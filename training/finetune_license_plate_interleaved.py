"""Second fine-tuning strategy for the "license_plate" 81st class, alongside
(not replacing) finetune_license_plate.py's frozen-weights approach.

finetune_license_plate.py prevents catastrophic forgetting of the other 80
COCO classes by freezing everything except the license_plate row of the
Detect head's cv3 branches (freeze=21 + per-row gradient hooks) - this
guarantees the other classes' weights are byte-identical before/after
training, but after 50 epochs the plate class itself still scored
0 precision/recall/mAP: a completely frozen backbone/neck has no way to
learn plate-specific features, and the license_plate-only training images
(cars in dashcam close-ups) are a narrow, unrepresentative slice of scenes
for the classes that *are* trainable.

This script instead trains the entire network (no freezing at all) and
protects the other 80 classes a different way: interleaving generic
COCO128 images (no plates, standard 80-class labels) into every training
batch, at roughly 1 non-plate image per `--interleave-ratio` plate images
(sensible range 5-20 per the request that prompted this script). The
model keeps seeing ordinary, class-balanced scenes throughout training,
so it has continued pressure to keep detecting the other 80 classes
correctly, rather than relying on gradients simply never reaching most of
the network.

Ratio guarantee: the combined manifest inserts the ratio deterministically
(every Nth line), but ultralytics' default DataLoader shuffles per epoch,
so the *expected* fraction of non-plate images per batch matches the
ratio - it is not a hard per-batch guarantee. Given the dataset sizes here
(hundreds vs. batch=16), the actual per-batch composition should track the
target ratio closely enough in practice; this is standard manifest-level
class balancing, not a custom weighted sampler.

The other lever this script exposes is `--plate-class-weight`: ultralytics
8.4.105's v8DetectionLoss already supports a `model.class_weights` tensor
(multiplies the per-class BCE classification loss - see
ultralytics/utils/loss.py) - this sets it to `plate_class_weight` for
license_plate and 1.0 for every other class, so recall on the rare plate
class can be pushed up without touching per-class freezing. Both knobs are
meant to be tuned together (by hand or via optuna) against the tradeoff
this script exists to make explicit: plate recall vs. other-class
accuracy, no longer resolved by construction the way frozen weights did.

Usage: python training/finetune_license_plate_interleaved.py \
    [--epochs 50] [--imgsz 256] [--interleave-ratio 10] \
    [--plate-class-weight 5.0]
"""
import argparse
import random
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

from finetune_license_plate import (
    CAR_CLASS_IDX,
    NEW_CLASS_IDX,
    NUM_ORIGINAL_CLASSES,
    _cv3_layers,
    restore_attention_position_encoding_bias,
)

REPO_ROOT = Path(__file__).parent.parent
DATASET_DIR = REPO_ROOT / "data" / "license_plates"
GENERATED_MANIFEST = DATASET_DIR / "train_interleaved.generated.txt"
GENERATED_DATASET_YAML = DATASET_DIR / "dataset_interleaved.generated.yaml"


def _list_images(source) -> list[Path]:
    """Resolve a dataset 'train'/'val' value (dir, .txt manifest, or list of
    either) to a flat list of absolute image paths, same three forms
    ultralytics itself accepts (see coco128.yaml's own header comment).
    """
    if isinstance(source, list):
        images: list[Path] = []
        for item in source:
            images.extend(_list_images(item))
        return images

    p = Path(source)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted(x.resolve() for x in p.rglob("*") if x.suffix.lower() in exts)
    if p.is_file() and p.suffix == ".txt":
        base = p.parent
        images = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            lp = Path(line)
            images.append(lp.resolve() if lp.is_absolute() else (base / lp).resolve())
        return images
    raise ValueError(f"Don't know how to list images from {source!r}")


def build_interleaved_manifest(ratio: int, seed: int) -> tuple[int, int, int]:
    """Write GENERATED_MANIFEST: every plate training image, with one
    COCO128 (non-plate) image inserted after every `ratio` plate images.
    Downloads COCO128 on first use (ultralytics' own auto-download, ~7MB,
    cached under the ultralytics datasets_dir - see `yolo settings`).
    Returns (num_plate_images, num_nonplate_images, num_lines_written).
    """
    plate_images = _list_images(DATASET_DIR / "train.txt")

    coco128_info = check_det_dataset("coco128.yaml")
    nonplate_pool = _list_images(coco128_info["train"])
    if not nonplate_pool:
        raise RuntimeError("COCO128 train split resolved to zero images")

    rng = random.Random(seed)
    rng.shuffle(nonplate_pool)

    combined: list[Path] = []
    pool_i = 0
    for i, img in enumerate(plate_images, start=1):
        combined.append(img)
        if i % ratio == 0:
            combined.append(nonplate_pool[pool_i % len(nonplate_pool)])
            pool_i += 1

    GENERATED_MANIFEST.write_text("\n".join(str(p) for p in combined) + "\n")
    return len(plate_images), pool_i, len(combined)


def write_interleaved_dataset_yaml(base_yaml: Path) -> Path:
    """Copy dataset.yaml with `train:` repointed at GENERATED_MANIFEST
    (absolute path, so it's used as-is regardless of `path:`). `val:`
    stays the untouched plate-only validation set, so mAP/recall on
    license_plate remains directly comparable across runs/scripts.
    """
    config = yaml.safe_load(base_yaml.read_text())
    config["train"] = str(GENERATED_MANIFEST.resolve())
    GENERATED_DATASET_YAML.write_text(yaml.safe_dump(config, sort_keys=False))
    return GENERATED_DATASET_YAML


def _warm_start_row(p_conv, p_bn, l_conv, l_bn, kind):
    """Initialize the license_plate row (and restore old-class rows to
    their exact pretrained values) - unlike finetune_license_plate.py's
    _restore_and_guard, this registers no gradient hooks and leaves BN
    momentum alone: every row is free to train from here, this only sets
    a sane starting point.
    """
    with torch.no_grad():
        if kind == "depthwise":
            l_conv.weight[:NUM_ORIGINAL_CLASSES] = p_conv.weight
            l_conv.weight[NEW_CLASS_IDX] = p_conv.weight[CAR_CLASS_IDX] * 0.5
        else:
            in_old = p_conv.weight.shape[1]
            in_new = l_conv.weight.shape[1]
            l_conv.weight[:NUM_ORIGINAL_CLASSES, :in_old] = p_conv.weight
            l_conv.weight[NEW_CLASS_IDX, :in_old] = p_conv.weight[CAR_CLASS_IDX] * 0.5
            if in_new > in_old:
                l_conv.weight[:NUM_ORIGINAL_CLASSES, in_old:] = 0
                l_conv.weight[NEW_CLASS_IDX, in_old:] = 0

        if l_conv.bias is not None:
            l_conv.bias[:NUM_ORIGINAL_CLASSES] = p_conv.bias
            l_conv.bias[NEW_CLASS_IDX] = p_conv.bias[CAR_CLASS_IDX] - 2.0

        if l_bn is not None and p_bn is not None:
            l_bn.weight[:NUM_ORIGINAL_CLASSES] = p_bn.weight
            l_bn.bias[:NUM_ORIGINAL_CLASSES] = p_bn.bias
            l_bn.running_mean[:NUM_ORIGINAL_CLASSES] = p_bn.running_mean
            l_bn.running_var[:NUM_ORIGINAL_CLASSES] = p_bn.running_var
            l_bn.weight[NEW_CLASS_IDX] = p_bn.weight[CAR_CLASS_IDX]
            l_bn.bias[NEW_CLASS_IDX] = p_bn.bias[CAR_CLASS_IDX]
            l_bn.running_mean[NEW_CLASS_IDX] = p_bn.running_mean[CAR_CLASS_IDX]
            l_bn.running_var[NEW_CLASS_IDX] = p_bn.running_var[CAR_CLASS_IDX]
            # momentum left at its default - running stats update normally,
            # same as every other class, since nothing here is frozen.


def prepare_class_head_and_weights(trainer, base_model_path: str, plate_class_weight: float):
    """on_train_start callback: warm-start the license_plate row (see
    _warm_start_row), fix the same attn.pe.conv bias reconstruction bug
    finetune_license_plate.py works around (unrelated to freezing, still
    applies here), and set model.class_weights so v8DetectionLoss
    up-weights the plate class's classification loss - see module
    docstring for why this replaces per-row gradient freezing here.
    """
    restore_attention_position_encoding_bias(trainer.model, base_model_path)

    pretrained = YOLO(base_model_path)
    pretrained_head = pretrained.model.model[-1]
    live_head = trainer.model.model[-1]
    assert live_head.nc == NUM_ORIGINAL_CLASSES + 1, f"expected nc=81, got {live_head.nc}"

    for pretrained_branch, live_branch in zip(pretrained_head.cv3, live_head.cv3):
        pretrained_layers = _cv3_layers(pretrained_branch)
        live_layers = _cv3_layers(live_branch)
        for (p_conv, p_bn, kind), (l_conv, l_bn, _) in zip(pretrained_layers, live_layers):
            if kind == "freeze":
                continue  # fixed-width (64ch) layer, not nc-indexed - nothing to warm-start
            _warm_start_row(p_conv, p_bn, l_conv, l_bn, kind)

    class_weights = torch.ones(NUM_ORIGINAL_CLASSES + 1)
    class_weights[NEW_CLASS_IDX] = plate_class_weight
    trainer.model.class_weights = class_weights

    print(
        f"[finetune-interleaved] Warm-started license_plate row across "
        f"{len(pretrained_head.cv3)} cv3 branches, class_weights[license_plate]={plate_class_weight}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="yolo12n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Layers to freeze (default 0 - the whole point of this script is training everything).",
    )
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument(
        "--interleave-ratio",
        type=int,
        default=10,
        help="1 non-plate COCO128 image inserted per N plate images (sensible range 5-20).",
    )
    parser.add_argument(
        "--plate-class-weight",
        type=float,
        default=5.0,
        help="Classification-loss multiplier for the license_plate class (1.0 = no boost).",
    )
    parser.add_argument("--interleave-seed", type=int, default=0)
    parser.add_argument("--data", default=str(DATASET_DIR / "dataset.yaml"))
    parser.add_argument("--project", default="runs/license_plate")
    parser.add_argument("--name", default="license_plate_ft_interleaved")
    args = parser.parse_args()

    if not 5 <= args.interleave_ratio <= 20:
        print(
            f"[finetune-interleaved] Note: --interleave-ratio {args.interleave_ratio} "
            "is outside the 5-20 range this script was designed around."
        )

    print(f"[finetune-interleaved] Building interleaved manifest (ratio=1:{args.interleave_ratio})")
    n_plate, n_nonplate, n_total = build_interleaved_manifest(args.interleave_ratio, args.interleave_seed)
    print(
        f"[finetune-interleaved] {n_plate} plate images + {n_nonplate} COCO128 images "
        f"= {n_total} total training images -> {GENERATED_MANIFEST}"
    )
    interleaved_data_yaml = write_interleaved_dataset_yaml(Path(args.data))

    print(f"[finetune-interleaved] Loading base model {args.base_model}")
    model = YOLO(args.base_model)
    names = dict(model.model.names)
    names[NEW_CLASS_IDX] = "license_plate"
    model.model.names = names
    model.model.yaml["nc"] = NUM_ORIGINAL_CLASSES + 1

    seed_path = "training/yolo12n_81class_seed.pt"
    model.save(seed_path)
    print(f"[finetune-interleaved] Saved seed checkpoint to {seed_path}")

    train_model = YOLO(seed_path)
    train_model.add_callback(
        "on_train_start",
        lambda trainer: prepare_class_head_and_weights(trainer, args.base_model, args.plate_class_weight),
    )

    print(
        f"[finetune-interleaved] Starting training: epochs={args.epochs} imgsz={args.imgsz} "
        f"freeze={args.freeze} plate_class_weight={args.plate_class_weight}"
    )
    train_model.train(
        data=str(interleaved_data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    best_path = str(train_model.trainer.save_dir / "weights" / "best.pt")
    print(f"[finetune-interleaved] Re-applying attn.pe bias fix to final checkpoint {best_path}")
    final_model = YOLO(best_path)
    restore_attention_position_encoding_bias(final_model.model, args.base_model)
    final_model.save(best_path)
    print(f"[finetune-interleaved] Done - {best_path} is ready to export")


if __name__ == "__main__":
    main()
