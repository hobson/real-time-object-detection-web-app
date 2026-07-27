"""Validates that the in-browser YOLO pipeline can recognize the object
classes a self-driving car needs to care about, using KITTI images.

This deliberately does NOT drive a real browser. `postprocess.py` is a
documented line-for-line port of `components/models/Yolo.tsx`'s
postprocessing (see its own module docstring), and this script runs the
exact same `.onnx` files under `models/` with the exact same preprocessing
main.py's `/predict` uses - so "does postprocess.py + these weights
recognize X" is exactly "does the in-browser client recognize X", without
needing WASM/browser automation to prove it.

Dataset: data/external_datasets/kitti (Ultralytics' 8-class KITTI subset -
see kitti.yaml). Two label sets are used:
  - labels/val: the original KITTI object classes (car, van, truck,
    pedestrian, Person_sitting, cyclist, tram, misc).
  - labels_license_plate_only/val: machine-generated labels (an existing
    plate detector run once over the KITTI images, per this directory's
    name) - only ~184 of the 1496 val images have a visible plate.

Coverage vs. the "self-driving car" class list this was asked to validate:
  bicycle    -> KITTI's "cyclist" (a person ON a bicycle - COCO could
               reasonably answer with either/both "person" and "bicycle")
  motorcycle -> NOT COVERED. This KITTI subset has no motorcycle-labeled
               images at all (not a KITTI limitation exactly - Ultralytics'
               reduced 8-class set just doesn't include one). Reported as
               a known gap below rather than silently skipped.
  car        -> KITTI "car"
  all vehicles -> KITTI "car"/"van"/"truck"/"tram" (tram has no real COCO
               equivalent; matched leniently against "train"/"bus")
  pedestrians -> KITTI "pedestrian"
  occupants of parked cars -> KITTI "Person_sitting", which the official
               KITTI benchmark defines as a sitting person typically
               partially occluded by/inside a vehicle - the closest
               available proxy for this requirement.
  license plates -> labels_license_plate_only, matched against the
               separate single-class plate detector
               (yolo-v9-t-384-license-plate-end2end.onnx), run at the same
               384x384 resolution Yolo.tsx uses for it.

Usage (from inference-server/, same venv as main.py):
    python test_kitti_detection.py
    python test_kitti_detection.py --num-images 300 --iou-threshold 0.3
    # Compare a higher-resolution model - KITTI's wide 1242x375 aspect
    # ratio gets squashed a lot at 256x256 (see finding below):
    python test_kitti_detection.py --object-model yolov7-tiny_640x640.onnx --object-resolution 640 640

Known finding from running this (yolo12n.onnx @ 256x256, 300 val images,
IoU>=0.3): car recall 16.0%, pedestrian/cyclist/Person_sitting recall
0.0%, license_plate recall 0.0%. Confirmed NOT a bug in this script - the
live production server (main.py's actual /predict) returns the identical
empty detections for the same KITTI images. Switching to
yolov7-tiny_640x640.onnx (less aspect-ratio squashing) roughly doubles car
recall (16.0% -> 40.6%) but pedestrian/cyclist recall stays at 0.0% even
then - a real domain-gap finding: these COCO-trained nano/tiny models
struggle with KITTI-style dashcam imagery (distant, small, low-contrast
subjects from a car-mounted camera), especially for people. This is a
capability gap to be aware of, not something this test papers over with a
lenient threshold - a genuinely safety-critical self-driving-car use case
would need domain-specific fine-tuning (e.g. on KITTI or similar driving
footage) rather than relying on these general-purpose COCO weights as-is.
"""
import argparse
from pathlib import Path

import onnxruntime as ort
from PIL import Image

from postprocess import POSTPROCESS_MAP, postprocess_yolov7, preprocess

REPO_ROOT = Path(__file__).parent.parent
KITTI_DIR = REPO_ROOT / "data" / "external_datasets" / "kitti"
MODELS_DIR = REPO_ROOT / "models"

OBJECT_MODEL = "yolo12n.onnx"
OBJECT_RESOLUTION = (256, 256)
PLATE_MODEL = "yolo-v9-t-384-license-plate-end2end.onnx"
PLATE_RESOLUTION = (384, 384)

KITTI_CLASSES = [
    "car", "van", "truck", "pedestrian", "Person_sitting", "cyclist",
    "tram", "misc",
]

# Which COCO classes count as a hit for each KITTI ground-truth class -
# see module docstring for the reasoning behind each mapping. "misc" is
# KITTI's own catch-all/ambiguous bucket and isn't checked against anything.
REQUIRED_COCO_MATCH = {
    "car": {"car"},
    "van": {"car", "truck"},
    "truck": {"truck", "bus"},
    "pedestrian": {"person"},
    "Person_sitting": {"person"},
    "cyclist": {"person", "bicycle"},
    "tram": {"train", "bus"},
}


def _load_session(model_name: str) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(MODELS_DIR / model_name), providers=["CPUExecutionProvider"]
    )


def _run_model(session: ort.InferenceSession, image: Image.Image, resolution: tuple[int, int], postprocess_fn) -> list[dict]:
    tensor = preprocess(image, resolution)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]
    return postprocess_fn(output, resolution)


def _yolo_txt_to_boxes(label_path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Parse a YOLO-format label file: `class_id x_center y_center w h`,
    all normalized 0-1. Returns (class_id, (x0, y0, x1, y1))."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls_id, xc, yc, w, h = (float(v) for v in line.split())
        boxes.append((int(cls_id), (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)))
    return boxes


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _has_match(gt_box, detections: list[dict], allowed_classes: set[str], iou_threshold: float) -> bool:
    return any(
        d["class"] in allowed_classes and _iou(gt_box, tuple(d["box"])) >= iou_threshold
        for d in detections
    )


def _evaluate_plates(
    image: Image.Image, plate_session: ort.InferenceSession, plate_label_path: Path, iou_threshold: float
) -> tuple[int, int, bool]:
    """Returns (gt_count, hit_count, had_label) for one image's plate
    ground truth, if any - a self-contained pass separate from the main
    object-detection loop in evaluate(), since it's a genuinely independent
    measurement (different model, different label set)."""
    plate_boxes = _yolo_txt_to_boxes(plate_label_path)
    if not plate_boxes:
        return 0, 0, False
    plate_detections = _run_model(plate_session, image, PLATE_RESOLUTION, postprocess_yolov7)
    hits = sum(
        _has_match(gt_box, plate_detections, {"license_plate"}, iou_threshold)
        for _cls_id, gt_box in plate_boxes
    )
    return len(plate_boxes), hits, True


def evaluate(
    num_images: int,
    iou_threshold: float,
    object_model: str = OBJECT_MODEL,
    object_resolution: tuple[int, int] = OBJECT_RESOLUTION,
    check_plates: bool = True,
) -> None:
    object_session = _load_session(object_model)
    plate_session = _load_session(PLATE_MODEL) if check_plates else None

    images_dir = KITTI_DIR / "images" / "val"
    labels_dir = KITTI_DIR / "labels" / "val"
    plate_labels_dir = KITTI_DIR / "labels_license_plate_only" / "val"

    image_paths = sorted(images_dir.glob("*.png"))[:num_images]
    print(f"Evaluating {len(image_paths)} KITTI val images "
          f"({object_model} @ {object_resolution}"
          + (f", {PLATE_MODEL} @ {PLATE_RESOLUTION}" if check_plates else "")
          + f", IoU >= {iou_threshold})\n")

    gt_counts = {c: 0 for c in KITTI_CLASSES}
    hit_counts = {c: 0 for c in KITTI_CLASSES}
    plate_gt = 0
    plate_hits = 0
    images_with_plate_labels = 0

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        object_detections = _run_model(
            object_session, image, object_resolution, POSTPROCESS_MAP[object_model]
        )

        for cls_id, gt_box in _yolo_txt_to_boxes(labels_dir / f"{image_path.stem}.txt"):
            cls_name = KITTI_CLASSES[cls_id]
            if cls_name not in REQUIRED_COCO_MATCH:
                continue
            gt_counts[cls_name] += 1
            if _has_match(gt_box, object_detections, REQUIRED_COCO_MATCH[cls_name], iou_threshold):
                hit_counts[cls_name] += 1

        if check_plates:
            gt, hits, had_label = _evaluate_plates(
                image, plate_session, plate_labels_dir / f"{image_path.stem}.txt", iou_threshold
            )
            plate_gt += gt
            plate_hits += hits
            images_with_plate_labels += had_label

    print(f"{'Class':<16}{'GT boxes':>10}{'Recall':>10}")
    print("-" * 36)
    for cls_name in KITTI_CLASSES:
        if cls_name not in REQUIRED_COCO_MATCH:
            continue
        n = gt_counts[cls_name]
        recall = hit_counts[cls_name] / n if n else float("nan")
        print(f"{cls_name:<16}{n:>10}{'n/a' if n == 0 else f'{recall:.1%}':>10}")
    print("-" * 36)
    if check_plates:
        plate_recall = plate_hits / plate_gt if plate_gt else float("nan")
        print(f"{'license_plate':<16}{plate_gt:>10}{'n/a' if plate_gt == 0 else f'{plate_recall:.1%}':>10}"
              f"  ({images_with_plate_labels} images had a plate label)")

    print(
        "\nKnown gap: this KITTI subset (Ultralytics' 8-class kitti.yaml) has no "
        "motorcycle-labeled images, so motorcycle recall isn't measured here - "
        "supplement with another dataset if that specifically needs coverage."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-images", type=int, default=300, help="Number of KITTI val images to evaluate")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="Minimum IoU to count as a detection match")
    parser.add_argument(
        "--object-model", default=OBJECT_MODEL,
        help="Any RES_TO_MODEL entry from Yolo.tsx (e.g. yolov7-tiny_640x640.onnx) - "
             "KITTI's wide 1242x375 images are heavily squashed at low resolutions, "
             "so this is worth comparing across.",
    )
    parser.add_argument(
        "--object-resolution", type=int, nargs=2, default=list(OBJECT_RESOLUTION), metavar=("W", "H"),
        help="Must match --object-model's own input resolution (see Yolo.tsx's RES_TO_MODEL)",
    )
    parser.add_argument(
        "--no-plates", action="store_true",
        help="Skip the license-plate model pass (only evaluate --object-model)",
    )
    args = parser.parse_args()
    evaluate(
        args.num_images, args.iou_threshold,
        object_model=args.object_model,
        object_resolution=tuple(args.object_resolution),
        check_plates=not args.no_plates,
    )


if __name__ == "__main__":
    main()
