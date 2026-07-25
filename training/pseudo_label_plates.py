"""Pseudo-label the license plate dataset's other 80 COCO classes.

The 724 images in data/license_plates/ are only manually annotated for
license_plate - but nearly every one also contains a car, often a person,
sometimes street signs, etc. Fine-tuning on plate-only labels teaches the
model those other real, visible objects are false positives (the loss
sees an unlabeled car in the image and penalizes detecting it) - the
gradient-freezing machinery in finetune_license_plate.py defends against
this, but a cleaner, complementary fix is to just... label them, using
the pretrained model itself as the labeler.

This is NOT the same as training on the pseudo-labels blindly - it's
standard-issue noisy pseudo-labeling (the pretrained model's own
mistakes become label noise), so a confidence threshold is applied, and
this should be treated as a precision/recall trade against catastrophic
forgetting, not ground truth. Spot-check a sample after running.

Usage: python training/pseudo_label_plates.py [--conf 0.25]

Idempotent: always regenerates from data/license_plates/labels_plates_only/
(the untouched manual license_plate-only annotations) rather than reading
its own prior output, so re-running doesn't accumulate duplicate boxes.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).parent.parent
DATASET_DIR = REPO_ROOT / "data" / "license_plates"
PLATES_ONLY_LABELS = DATASET_DIR / "labels_plates_only"
IMAGES_DIR = DATASET_DIR / "images"
OUTPUT_LABELS = DATASET_DIR / "labels"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo12n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=256)
    args = parser.parse_args()

    if not PLATES_ONLY_LABELS.exists():
        raise SystemExit(
            f"{PLATES_ONLY_LABELS} doesn't exist - expected the untouched "
            "plate-only annotations there. Refusing to guess; back up "
            "labels/ there first if this is a fresh checkout."
        )

    model = YOLO(args.model)
    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"[pseudo-label] {len(image_paths)} images, conf>={args.conf}")

    total_pseudo_boxes = 0
    class_counts: dict[str, int] = {}

    for image_path in image_paths:
        stem = image_path.stem
        plate_label_path = PLATES_ONLY_LABELS / f"{stem}.txt"
        plate_lines = (
            plate_label_path.read_text().strip().splitlines()
            if plate_label_path.exists()
            else []
        )

        results = model.predict(
            str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False
        )
        pseudo_lines = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                cx, cy, w, h = box.xywhn[0].tolist()
                pseudo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                total_pseudo_boxes += 1
                class_counts[model.names[cls_id]] = (
                    class_counts.get(model.names[cls_id], 0) + 1
                )

        combined = plate_lines + pseudo_lines
        (OUTPUT_LABELS / f"{stem}.txt").write_text("\n".join(combined) + "\n")

    print(f"[pseudo-label] Added {total_pseudo_boxes} pseudo-labels across {len(image_paths)} images")
    print("[pseudo-label] By class:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls_name}: {count}")


if __name__ == "__main__":
    main()
