"""DB-driven replacement for train_resume_all_categories.py's local-file
dataset construction - everything downstream (class_weights mechanism, the
attn.pe bias fix, no freezing) is unchanged; only where training images/
labels come from and which ones get used is different.

Two things this adds that a flat YOLO-file dataset structurally can't
represent (see orm.py's label_source/cross_model_agreement columns and
compute_label_confidence.py, which populates them):

1. **Label-quality filtering.** Only "trusted" labels are exported into the
   training set - label_source="human_dataset" (real ground truth), or a
   machine label whose cross_model_agreement is at least
   TRUSTED_AGREEMENT_THRESHOLD (independent reference models corroborate
   it). Everything else is left out of that image's label file entirely -
   directly addressing the hypothesis that some of this training set's
   labels are simply wrong, rather than training on them and hoping
   per-class reweighting alone compensates.

2. **Oversampling trusted-but-currently-wrong examples.** For each image,
   if any trusted label has no matching detection (IoU>=0.5, same class)
   among --current-model-name's own already-persisted fresh detections on
   that image (see compute_label_confidence.py, which ran and stored the
   current checkpoint's detections for exactly this purpose), that image
   is repeated --oversample-repeats times in the generated train manifest.
   This is "well-labeled examples the model currently gets wrong" -
   precisely the signal that was missing when license_plate's accuracy
   regressed across rounds 2->3 despite already being up-weighted near the
   ceiling (see docs/... this session's investigation).

Caveat: --current-model-name must match whatever checkpoint compute_label_
confidence.py most recently ran and persisted detections for. Re-run that
script against a fresh checkpoint before each new train_from_db.py round if
you want the oversampling signal to reflect that round's actual mistakes,
rather than reusing a stale prior round's.

Val set is unchanged from train_resume_all_categories.py (the same broad
KITTI-val + license_plates-val file-based manifest) - kept file-based, not
DB-based, so Fbeta stays directly comparable across every round this
session's iterative_reweight_train.py loop has already run.

Usage: python training/train_from_db.py \
    [--checkpoint runs/.../license_plate_ft_reweight_round3/weights/best.pt] \
    [--current-model-name license_plate_ft_reweight_round3] \
    [--epochs 50] [--agreement-threshold 0.67] [--oversample-repeats 3]
"""
import argparse
import random
import sys
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

REPO_ROOT = Path(__file__).parent.parent
INFERENCE_SERVER_DIR = REPO_ROOT / "inference-server"
TRAINING_DIR = REPO_ROOT / "training"

load_dotenv(INFERENCE_SERVER_DIR / ".env")
sys.path.insert(0, str(INFERENCE_SERVER_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from orm import Image, engine_from_env  # noqa: E402
from review_confusion_matrix import iou  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from device_check import resolve_device  # noqa: E402
from finetune_license_plate import restore_attention_position_encoding_bias  # noqa: E402
from train_resume_all_categories import (  # noqa: E402
    DATASET_DIR, GENERATED_VAL_MANIFEST, build_broad_val_manifest,
    compute_balanced_class_weights, label_path_for_image,
)

GENERATED_DIR = REPO_ROOT / "data" / "external_datasets" / "db_export"
GENERATED_TRAIN_MANIFEST = GENERATED_DIR / "train.generated.txt"
GENERATED_DATASET_YAML = GENERATED_DIR / "dataset.generated.yaml"

TRUSTED_AGREEMENT_THRESHOLD = 0.67  # >= 2 of 3 reference models corroborate
IOU_MATCH_THRESHOLD = 0.5


def _to_xyxy(label) -> tuple[float, float, float, float]:
    return (
        label.x_center - label.width / 2, label.y_center - label.height / 2,
        label.x_center + label.width / 2, label.y_center + label.height / 2,
    )


def _model_missed(label, current_model_detections: list[tuple[int, tuple]]) -> bool:
    label_box = _to_xyxy(label)
    return not any(
        cls_id == label.class_id and iou(label_box, box) >= IOU_MATCH_THRESHOLD
        for cls_id, box in current_model_detections
    )


def build_db_manifest(
    session: Session, current_model_name: str, agreement_threshold: float, oversample_repeats: int, fraction: float,
) -> tuple[int, int, int]:
    """Writes GENERATED_DIR/{images,labels}/<id>.{ext,txt} (images as
    symlinks to the originals, labels as trusted-only YOLO label files) and
    GENERATED_TRAIN_MANIFEST (with oversample repeats). Returns (n_images,
    n_oversampled, n_manifest_lines)."""
    images_dir = GENERATED_DIR / "images"
    labels_dir = GENERATED_DIR / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    all_images = session.query(Image).filter(Image.training_status == Image.TRAINING_STATUS_APPROVED).all()
    rng = random.Random(0)
    if fraction < 1.0:
        rng.shuffle(all_images)
        all_images = all_images[: max(1, int(len(all_images) * fraction))]

    manifest: list[str] = []
    n_images = 0
    n_oversampled = 0
    for image in all_images:
        labels = image.annotations
        trusted = [l for l in labels if l.is_trusted(TRUSTED_AGREEMENT_THRESHOLD)]
        if not trusted:
            continue  # nothing worth training on for this image

        src = Path(image.file_path)
        link_path = images_dir / f"{image.id}{src.suffix}"
        if not link_path.exists():
            link_path.symlink_to(src.resolve())
        label_path = labels_dir / f"{image.id}.txt"
        label_path.write_text(
            "\n".join(f"{l.class_id} {l.x_center} {l.y_center} {l.width} {l.height}" for l in trusted) + "\n"
        )

        current_model_detections = [
            (l.class_id, _to_xyxy(l)) for l in labels
            if l.label_source.is_model(current_model_name)
        ]
        model_gets_some_wrong = any(_model_missed(l, current_model_detections) for l in trusted)

        # str(link_path), NOT link_path.resolve() - resolving would follow
        # the symlink to its real target (inference-server/storage/images/
        # <sha256>.ext), and ultralytics derives each label path from the
        # IMAGE path by replacing an "images" path segment with "labels"
        # (img2label_paths) - it needs to see GENERATED_DIR/images/<id>.ext
        # (whose sibling GENERATED_DIR/labels/<id>.txt we just wrote), not
        # the original storage path, which has no such per-id label file.
        repeats = oversample_repeats if model_gets_some_wrong else 1
        manifest.extend([str(link_path)] * repeats)
        n_images += 1
        n_oversampled += model_gets_some_wrong

    rng.shuffle(manifest)
    GENERATED_TRAIN_MANIFEST.write_text("\n".join(manifest) + "\n")
    return n_images, n_oversampled, len(manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Starting checkpoint - a previously fine-tuned 81(+)-class model.")
    parser.add_argument("--base-model", default="yolo12n.pt", help="Used only for the attn.pe bias-fix reference.")
    parser.add_argument(
        "--current-model-name", default="license_plate_ft_reweight_round3",
        help="Must match the model_name compute_label_confidence.py most recently persisted detections under - see module docstring.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=0.0005)
    parser.add_argument(
        "--device", default="cpu", type=resolve_device,
        help="Passed to model.train(device=...) - verified via device_check.resolve_device as part of "
        "argument parsing itself (type=resolve_device), so a non-cpu request is smoke-tested before use "
        "and falls back to cpu if it fails, with no separate call a future edit could forget. Default 'cpu'.",
    )
    parser.add_argument(
        "--cos-lr", action="store_true",
        help="Use cosine LR decay instead of ultralytics' default linear decay - may help stabilize "
        "convergence over many epochs. Off by default to preserve existing behavior.",
    )
    parser.add_argument(
        "--warmup-epochs", type=float, default=3.0,
        help="ultralytics default is 3.0 epochs of LR warmup - appropriate when training from "
        "scratch/COCO weights, but every round here fine-tunes from an already-good checkpoint "
        "(--checkpoint), so a long warmup may waste early epochs at a below-target LR instead of "
        "adapting to this round's reweighted loss. Consider passing 0-1 to test faster adaptation.",
    )
    parser.add_argument(
        "--multi-scale", action="store_true",
        help="Randomly vary input resolution during training (ultralytics' multi_scale) - may "
        "improve robustness to license plates photographed at different distances/scales. Off by "
        "default to preserve existing behavior.",
    )
    parser.add_argument("--agreement-threshold", type=float, default=TRUSTED_AGREEMENT_THRESHOLD)
    parser.add_argument("--oversample-repeats", type=int, default=3)
    parser.add_argument("--fraction", type=float, default=1.0, help="Subsample of DB images (for smoke-testing).")
    parser.add_argument("--class-weights-file", default=None)
    parser.add_argument("--project", default="runs/license_plate")
    parser.add_argument("--name", default="license_plate_ft_from_db")
    parser.add_argument("--best-path-out", default=None)
    args = parser.parse_args()

    dataset_config = yaml.safe_load((DATASET_DIR / "dataset.yaml").read_text())
    num_classes = len(dataset_config["names"])

    engine = engine_from_env()
    with Session(engine) as session:
        print(f"[train-from-db] Building DB-driven manifest (agreement_threshold={args.agreement_threshold}, oversample_repeats={args.oversample_repeats})")
        n_images, n_oversampled, n_lines = build_db_manifest(
            session, args.current_model_name, args.agreement_threshold, args.oversample_repeats, args.fraction,
        )
    print(f"[train-from-db] {n_images} images with trusted labels, {n_oversampled} oversampled ({n_oversampled/n_images:.1%}), {n_lines} manifest lines")

    all_images = [Path(line) for line in GENERATED_TRAIN_MANIFEST.read_text().splitlines() if line.strip()]
    if args.class_weights_file:
        class_weights = torch.load(args.class_weights_file)
        print(f"[train-from-db] Loaded class_weights from {args.class_weights_file}")
    else:
        class_weights = compute_balanced_class_weights(all_images, num_classes)

    print("[train-from-db] Building broad val manifest (KITTI val split + license_plates val) - unchanged from train_resume_all_categories.py, for cross-round comparability")
    build_broad_val_manifest()

    config = dict(dataset_config)
    config["train"] = str(GENERATED_TRAIN_MANIFEST.resolve())
    config["val"] = str(GENERATED_VAL_MANIFEST.resolve())
    GENERATED_DATASET_YAML.write_text(yaml.safe_dump(config, sort_keys=False))

    print(f"[train-from-db] Loading checkpoint {args.checkpoint}")
    model = YOLO(args.checkpoint)
    model.add_callback("on_train_start", lambda trainer: setattr(trainer.model, "class_weights", class_weights))

    print(f"[train-from-db] Starting training: epochs={args.epochs} imgsz={args.imgsz} freeze={args.freeze} device={args.device}")
    model.train(
        data=str(GENERATED_DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        lr0=args.lr0,
        cos_lr=args.cos_lr,
        warmup_epochs=args.warmup_epochs,
        multi_scale=args.multi_scale,
        project=args.project,
        name=args.name,
        exist_ok=True,
        device=args.device,
    )

    best_path = str(model.trainer.save_dir / "weights" / "best.pt")
    print(f"[train-from-db] Re-applying attn.pe bias fix to final checkpoint {best_path}")
    final_model = YOLO(best_path)
    restore_attention_position_encoding_bias(final_model.model, args.base_model)
    final_model.save(best_path)
    print(f"[train-from-db] Done - {best_path} is ready to export")

    if args.best_path_out:
        Path(args.best_path_out).write_text(best_path)


if __name__ == "__main__":
    main()
