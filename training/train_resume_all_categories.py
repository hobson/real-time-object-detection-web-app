"""Continued training on a combined, all-categories dataset: KITTI (a large
subset, remapped from its own 8-class scheme into this project's 81-class
one) + COCO128, both already carrying license_plate labels (see
docs/fine-tune-yolo-on-new-category-interleaved.md's "COCO128/KITTI
actually contain some real plates" section - training/label_plates_via_api.py
found and added them). Resumes from a previously fine-tuned checkpoint
(default: the interleaved run's best.pt) rather than the original
pretrained yolo12n.pt.

Unlike finetune_license_plate_interleaved.py, this script:
- Applies NO manual class weight for license_plate specifically - instead
  computes a `class_weights='balanced'` tensor (sklearn's standard
  n_samples / (n_classes * n_samples_c) formula) from actual per-class
  instance counts across the combined training set, same treatment for
  every class including license_plate.
- Freezes nothing (freeze=0), same as the interleaved script.
- Does not warm-start/restore the Detect head's cv3 rows - the starting
  checkpoint already has a properly trained 81-class head from the prior
  run, so that surgery (needed only when a class is first introduced) does
  not apply here.

KITTI class remapping (KITTI_TO_UNIFIED below) is a judgment call, not a
verified ground truth - see its inline comments for the reasoning and
caveats on each mapping (van->car, cyclist->bicycle, tram->train, misc
dropped entirely).

Usage: python scripts/train_resume_all_categories.py \
    [--checkpoint runs/detect/runs/license_plate/license_plate_ft_interleaved/weights/best.pt] \
    [--epochs 50] [--batch 16] [--kitti-fraction 0.5]
"""
import argparse
import random
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))
from finetune_license_plate import (  # noqa: E402
    NEW_CLASS_IDX,
    restore_attention_position_encoding_bias,
)

DATASET_DIR = REPO_ROOT / "data" / "license_plates"
KITTI_DIR = REPO_ROOT / "data" / "external_datasets" / "kitti"
KITTI_UNIFIED_DIR = REPO_ROOT / "data" / "external_datasets" / "kitti_unified"
COCO128_DIR = REPO_ROOT / "data" / "external_datasets" / "coco128"
GENERATED_MANIFEST = REPO_ROOT / "data" / "external_datasets" / "all_categories_train.generated.txt"
GENERATED_DATASET_YAML = REPO_ROOT / "data" / "external_datasets" / "all_categories_dataset.generated.yaml"

# KITTI's own 8 classes (see data/external_datasets/kitti/kitti.yaml) mapped
# onto this project's 81-class scheme (COCO's 0-79 + license_plate=80, see
# data/license_plates/dataset.yaml). None = drop (no reasonable match).
#
#   0 car            -> 2  car
#   1 van             -> 2  car             (closer to "car" than "truck" for
#                                             most KITTI vans - passenger
#                                             minivans, not cargo trucks;
#                                             genuinely ambiguous either way)
#   2 truck           -> 7  truck
#   3 pedestrian      -> 0  person
#   4 Person_sitting  -> 0  person
#   5 cyclist         -> 1  bicycle         (KITTI's box includes the rider,
#                                             COCO's "bicycle" box normally
#                                             doesn't - imperfect but the
#                                             closest single-class match;
#                                             no COCO class combines both)
#   6 tram            -> 6  train           (closest rail-vehicle analog)
#   7 misc            -> None (dropped)      (too heterogeneous - construction
#                                             equipment, trailers, etc. - no
#                                             single COCO class fits)
KITTI_TO_UNIFIED = {0: 2, 1: 2, 2: 7, 3: 0, 4: 0, 5: 1, 6: 6, 7: None}


def remap_and_merge_kitti_labels(split: str) -> tuple[Path, Path]:
    """Builds data/external_datasets/kitti_unified/{images,labels}/<split>/
    - images/ as symlinks to the originals (so ultralytics' img2label_paths
    "images"->"labels" auto-derivation works without duplicating image
    bytes), labels/ as freshly remapped KITTI boxes + the license_plate
    boxes label_plates_via_api.py already found, merged into one file per
    image. Regenerated fresh every run (cheap - label files are tiny);
    never mutates the original kitti/ directory.
    """
    src_images = KITTI_DIR / "images" / split
    src_labels = KITTI_DIR / "labels" / split
    src_plate_labels = KITTI_DIR / "labels_license_plate_only" / split
    dst_images = KITTI_UNIFIED_DIR / "images" / split
    dst_labels = KITTI_UNIFIED_DIR / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    dropped_misc = 0
    total_boxes = 0
    n_images = 0
    for image_path in sorted(src_images.glob("*.png")) + sorted(src_images.glob("*.jpg")):
        n_images += 1
        link_path = dst_images / image_path.name
        if not link_path.exists():
            link_path.symlink_to(image_path.resolve())

        label_path = src_labels / f"{image_path.stem}.txt"
        lines = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                kitti_cid = int(parts[0])
                unified_cid = KITTI_TO_UNIFIED[kitti_cid]
                if unified_cid is None:
                    dropped_misc += 1
                    continue
                lines.append(" ".join([str(unified_cid), *parts[1:]]))

        plate_label_path = src_plate_labels / f"{image_path.stem}.txt"
        if plate_label_path.exists():
            for line in plate_label_path.read_text().splitlines():
                parts = line.split()
                if parts:
                    lines.append(" ".join([str(NEW_CLASS_IDX), *parts[1:]]))

        total_boxes += len(lines)
        (dst_labels / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    print(
        f"[train-resume] KITTI {split}: {n_images} images, {total_boxes} remapped+merged boxes, "
        f"dropped {dropped_misc} 'misc' boxes"
    )
    return dst_images, dst_labels


def build_combined_manifest(kitti_fraction: float, seed: int) -> tuple[int, int]:
    kitti_images_dir, _ = remap_and_merge_kitti_labels("train")
    kitti_images = sorted(kitti_images_dir.glob("*.png")) + sorted(kitti_images_dir.glob("*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(kitti_images)
    n_kitti = max(1, int(len(kitti_images) * kitti_fraction))
    kitti_subset = kitti_images[:n_kitti]

    coco128_images = sorted((COCO128_DIR / "images" / "train2017").glob("*.jpg"))

    all_images = [p.resolve() for p in kitti_subset] + [p.resolve() for p in coco128_images]
    rng.shuffle(all_images)
    GENERATED_MANIFEST.write_text("\n".join(str(p) for p in all_images) + "\n")
    return len(kitti_subset), len(coco128_images)


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def compute_balanced_class_weights(image_paths: list[Path], num_classes: int) -> torch.Tensor:
    """sklearn-style 'balanced' weighting: weight_c = n_total / (n_classes_present * n_c),
    computed from actual per-class box counts across the combined training
    set (not just KITTI or just COCO128 - the real mix this run trains on).
    Classes with zero instances get weight 1.0 (no gradient reaches them
    either way, so the weight is moot - just avoiding a div-by-zero).
    """
    counts = [0] * num_classes
    for image_path in image_paths:
        label_path = label_path_for_image(image_path)
        if not label_path.exists():
            continue
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if parts:
                counts[int(parts[0])] += 1

    total = sum(counts)
    n_present = sum(1 for c in counts if c > 0)
    weights = [total / (n_present * c) if c > 0 else 1.0 for c in counts]
    print(
        f"[train-resume] class_weights='balanced': {n_present}/{num_classes} classes present, "
        f"{total} total instances, license_plate weight={weights[NEW_CLASS_IDX]:.3f}"
    )
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="runs/detect/runs/license_plate/license_plate_ft_interleaved/weights/best.pt",
        help="Starting weights - a previously fine-tuned 81-class checkpoint, not the original pretrained model.",
    )
    parser.add_argument("--base-model", default="yolo12n.pt", help="Used only for the attn.pe bias-fix reference.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--freeze", type=int, default=0, help="Default 0 - nothing frozen, per the request this trains against.")
    parser.add_argument("--lr0", type=float, default=0.0005, help="Lower than a fresh fine-tune's default - this is continued training on an already-converged checkpoint.")
    parser.add_argument("--kitti-fraction", type=float, default=0.5, help="Fraction of KITTI's train split to sample in (a 'large subset', not all of it).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/license_plate")
    parser.add_argument("--name", default="license_plate_ft_all_categories")
    args = parser.parse_args()

    dataset_config = yaml.safe_load((DATASET_DIR / "dataset.yaml").read_text())
    num_classes = len(dataset_config["names"])

    print(f"[train-resume] Building combined KITTI(subset)+COCO128 manifest (kitti_fraction={args.kitti_fraction})")
    n_kitti, n_coco128 = build_combined_manifest(args.kitti_fraction, args.seed)
    all_images = [Path(line) for line in GENERATED_MANIFEST.read_text().splitlines() if line.strip()]
    print(f"[train-resume] {n_kitti} KITTI + {n_coco128} COCO128 = {len(all_images)} total training images")

    class_weights = compute_balanced_class_weights(all_images, num_classes)

    config = dict(dataset_config)
    config["train"] = str(GENERATED_MANIFEST.resolve())
    config["val"] = str((DATASET_DIR / "val.txt").resolve())
    GENERATED_DATASET_YAML.write_text(yaml.safe_dump(config, sort_keys=False))

    print(f"[train-resume] Loading checkpoint {args.checkpoint}")
    model = YOLO(args.checkpoint)
    model.add_callback(
        "on_train_start", lambda trainer: setattr(trainer.model, "class_weights", class_weights)
    )

    print(
        f"[train-resume] Starting training: epochs={args.epochs} imgsz={args.imgsz} "
        f"freeze={args.freeze} class_weights=balanced"
    )
    model.train(
        data=str(GENERATED_DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    best_path = str(model.trainer.save_dir / "weights" / "best.pt")
    print(f"[train-resume] Re-applying attn.pe bias fix to final checkpoint {best_path}")
    final_model = YOLO(best_path)
    restore_attention_position_encoding_bias(final_model.model, args.base_model)
    final_model.save(best_path)
    print(f"[train-resume] Done - {best_path} is ready to export")


if __name__ == "__main__":
    main()
