"""Auto-label license plates via the running inference-server's
`/alpr/predict` endpoint (fast-alpr's own YOLOv9 plate detector - see
inference-server/alpr.py), for two purposes:

1. Supplementing data/license_plates/labels_plates_only/ - those are Open
   Images V7's human annotations, which can miss a plate a plate-specific
   detector catches (e.g. a second vehicle in frame).
2. Labeling the non-plate datasets interleaved into training to guard
   against catastrophic forgetting (see
   docs/fine-tune-yolo-on-new-category-interleaved.md). Those datasets are
   used as "definitely no plate here" negative examples during interleaved
   training - if one of them actually contains a real, unlabeled plate,
   that silently contradicts the training signal (the model gets penalized
   for correctly detecting something that actually is there). This scans
   them and adds any plates found.

Never removes or edits existing lines - only appends new "<class_id> cx cy
w h" rows for detections that don't already overlap (IoU) an existing
same-class box in that image's label file, so re-running is idempotent and
existing ground truth is never overwritten.

Usage:
    # data/license_plates/ uses this project's 81-class scheme (COCO's
    # 0-79 + license_plate=80, see data/license_plates/dataset.yaml)
    python training/label_plates_via_api.py \\
        --images ../data/license_plates/images \\
        --labels ../data/license_plates/labels_plates_only \\
        --class-id 80

    # COCO128 (interleaved training's non-plate images) uses that same
    # 81-class scheme - it's COCO's own 0-79 classes, which is exactly
    # what indices 0-79 mean here too. Find its cached location first:
    #   python -c "from ultralytics.data.utils import check_det_dataset; \\
    #       print(check_det_dataset('coco128.yaml')['train'])"
    python training/label_plates_via_api.py \\
        --images <that path>/images/train2017 \\
        --labels <that path>/labels/train2017 \\
        --class-id 80

    # KITTI has its own unrelated 8-class scheme (car/van/truck/pedestrian/
    # ... - see data/external_datasets/kitti/kitti.yaml) that hasn't been
    # remapped into this project's 81-class scheme yet. Appending class-80
    # lines into KITTI's own label files would be syntactically fine (class
    # 80 is unused in an 8-class dataset) but semantically wrong to mix in
    # before that remap happens (KITTI's own class 0 means "car", not
    # "person" the way this project's class 0 does) - so plate detections
    # go to a SEPARATE directory instead, ready to merge in once someone
    # does that remap:
    python training/label_plates_via_api.py \\
        --images ../data/external_datasets/kitti/images/train \\
        --labels ../data/external_datasets/kitti/labels_license_plate_only/train \\
        --class-id 0

Requires the inference-server running first: cd inference-server && uvicorn main:app
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

Box = tuple[float, float, float, float]


def iou(box_a: Box, box_b: Box) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float) -> Box:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def load_existing_boxes(label_path: Path, class_id: int) -> list[Box]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5 and int(parts[0]) == class_id:
            boxes.append(yolo_to_xyxy(*(float(x) for x in parts[1:5])))
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images", required=True)
    parser.add_argument(
        "--labels", required=True,
        help="Output label dir - existing files are appended to, never overwritten wholesale",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8123/alpr/predict")
    parser.add_argument("--class-id", type=int, default=80)
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Min detection confidence to accept (higher than the app's usual 0.25 - "
        "these are out-of-domain/negative-example images, prioritize precision)",
    )
    parser.add_argument(
        "--iou-dedup", type=float, default=0.5,
        help="Skip a detection that overlaps an existing same-class box above this IoU",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after N images (0 = all)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        sys.exit(f"No images found in {images_dir}")

    health_url = args.url.rsplit("/", 1)[0] + "/health"
    try:
        requests.get(health_url, timeout=5).raise_for_status()
    except requests.RequestException as e:
        sys.exit(
            f"Inference server not reachable at {health_url}: {e}\n"
            "Start it first: cd inference-server && uvicorn main:app"
        )

    added_images = 0
    added_boxes = 0
    skipped_low_conf = 0
    skipped_dedup = 0
    errors = 0

    for i, image_path in enumerate(image_paths, start=1):
        try:
            resp = requests.post(args.url, data=image_path.read_bytes(), timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[{i}/{len(image_paths)}] {image_path.name}: request failed ({e})")
            errors += 1
            continue

        detections = resp.json().get("detections", [])
        label_path = labels_dir / f"{image_path.stem}.txt"
        existing_boxes = load_existing_boxes(label_path, args.class_id)
        new_lines = []

        for det in detections:
            if det["detectionConfidence"] < args.conf:
                skipped_low_conf += 1
                continue
            box = tuple(det["box"])
            if any(iou(box, b) > args.iou_dedup for b in existing_boxes):
                skipped_dedup += 1
                continue
            x0, y0, x1, y1 = box
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            new_lines.append(f"{args.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            existing_boxes.append(box)  # dedup against this run's own additions too

        if new_lines:
            with label_path.open("a") as f:
                for line in new_lines:
                    f.write(line + "\n")
            added_images += 1
            added_boxes += len(new_lines)
            print(f"[{i}/{len(image_paths)}] {image_path.name}: +{len(new_lines)} plate box(es)")

        if i % 200 == 0:
            print(f"...{i}/{len(image_paths)} processed")

    print(
        f"\nDone: {len(image_paths)} images scanned, {added_boxes} plate box(es) added "
        f"across {added_images} images. Skipped: {skipped_low_conf} below conf={args.conf}, "
        f"{skipped_dedup} already labeled (IoU>{args.iou_dedup}). Errors: {errors}."
    )


if __name__ == "__main__":
    main()
