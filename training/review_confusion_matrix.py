"""Visual per-class confusion-matrix review (first cut): for each requested
class, runs the trained model against a dataset split, IoU-matches
predictions to ground truth, and saves a single 2x2 grid image per class -
TP upper-left, FN upper-right, FP lower-left, TN lower-right (standard
confusion-matrix layout: rows = actual positive/negative, columns =
predicted positive/negative) - each cell showing one example image.

CAVEAT for anything other than license_plate: ground truth for the other
80 COCO classes in data/license_plates/ is machine-generated pseudo-labels
(see pseudo_label_plates.py's module docstring) - the pretrained model
labeling its own likely-correct guesses, confidence-thresholded, not
human-verified. FP/FN here for e.g. "car" or "person" partly reflects
pseudo-label noise, not only the fine-tuned model's real mistakes.
license_plate's ground truth (labels_plates_only/) IS manually annotated.

Quadrant definitions (per class, per image, via box-level IoU matching):
  TP: a predicted box of this class matches (IoU >= --iou-thresh) a
      ground-truth box of this class.
  FN: a ground-truth box of this class has no matching prediction (missed
      detection).
  FP: a predicted box of this class matches no ground-truth box (false
      alarm).
  TN: the image has zero ground-truth boxes AND zero predicted boxes of
      this class (correctly recognized absence).
First image found satisfying each quadrant is used - no attempt at
"best"/"most representative" selection yet.

Usage: python training/review_confusion_matrix.py \
    [--model runs/license_plate/license_plate_ft_interleaved/weights/best.pt] \
    [--classes license_plate,car,person] [--conf 0.25] [--iou-thresh 0.5] \
    [--split val] [--output runs/confusion_review]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from device_check import resolve_device

REPO_ROOT = Path(__file__).parent.parent
DATASET_DIR = REPO_ROOT / "data" / "license_plates"

COLORS = {
    "TP": (46, 204, 113),  # green
    "FN": (243, 156, 18),  # orange - ground truth the model missed
    "FP": (231, 76, 60),  # red - prediction with no matching ground truth
    "PLATE_CONTEXT": (52, 152, 219),  # blue - license_plate detections shown for context on non-plate classes' examples
}
# Standard confusion-matrix layout: rows = actual pos/neg, cols = predicted pos/neg
GRID_LAYOUT = {"TP": (0, 0), "FN": (0, 1), "FP": (1, 0), "TN": (1, 1)}
GRID_TITLES = {
    "TP": "True Positive",
    "FN": "False Negative (missed)",
    "FP": "False Positive",
    "TN": "True Negative",
}


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt_boxes: list, pred_boxes: list, iou_thresh: float):
    """Greedy IoU matching. Returns (matched_gt_idx, matched_pred_idx) sets."""
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    pairs = []
    for gi, gbox in enumerate(gt_boxes):
        for pi, pbox in enumerate(pred_boxes):
            score = iou(gbox, pbox)
            if score >= iou_thresh:
                pairs.append((score, gi, pi))
    for score, gi, pi in sorted(pairs, key=lambda t: -t[0]):
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
    return matched_gt, matched_pred


def parse_yolo_labels(label_path: Path, img_w: int, img_h: int) -> dict[int, list[tuple]]:
    """class_id -> list of (x0, y0, x1, y1) in pixel coords."""
    by_class: dict[int, list[tuple]] = {}
    if not label_path.exists():
        return by_class
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cid = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:5])
        x0 = (cx - w / 2) * img_w
        y0 = (cy - h / 2) * img_h
        x1 = (cx + w / 2) * img_w
        y1 = (cy + h / 2) * img_h
        by_class.setdefault(cid, []).append((x0, y0, x1, y1))
    return by_class


def label_path_for_image(image_path: Path) -> Path:
    """Mirrors ultralytics' own img2label_paths convention: replace the
    last "images" path segment with "labels", keep everything else,
    swap extension to .txt. Handles both this repo's flat
    data/license_plates/images/*.jpg layout and COCO128's nested
    coco128/images/train2017/*.jpg layout.
    """
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _load_font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_plate_context(draw: ImageDraw.ImageDraw, plate_boxes_conf, font):
    """Draw license_plate detections for spatial context on a non-plate
    class's example image (e.g. seeing where the model finds plates on a
    "car" example) - purely informational, not part of that class's own
    TP/FP/FN accounting.
    """
    for pbox, conf in plate_boxes_conf:
        x0, y0, x1, y1 = pbox
        draw.rectangle([x0, y0, x1, y1], outline=COLORS["PLATE_CONTEXT"], width=2)
        draw.text((x0 + 2, y1 - 14), f"plate {conf:.2f}", fill=COLORS["PLATE_CONTEXT"], font=font)


def render_annotated(
    image_path: Path, gt_boxes, pred_boxes_conf, tp_gt_idx, fp_pred_idx, fn_gt_idx, plate_boxes_conf=None
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font()

    if plate_boxes_conf:
        draw_plate_context(draw, plate_boxes_conf, font)
    for gi in fn_gt_idx:
        x0, y0, x1, y1 = gt_boxes[gi]
        draw.rectangle([x0, y0, x1, y1], outline=COLORS["FN"], width=3)
        draw.text((x0 + 2, y0 + 2), "FN (missed)", fill=COLORS["FN"], font=font)
    for gi in tp_gt_idx:
        x0, y0, x1, y1 = gt_boxes[gi]
        draw.rectangle([x0, y0, x1, y1], outline=COLORS["TP"], width=3)
        draw.text((x0 + 2, y0 + 2), "TP", fill=COLORS["TP"], font=font)
    for pi in fp_pred_idx:
        pbox, conf = pred_boxes_conf[pi]
        x0, y0, x1, y1 = pbox
        draw.rectangle([x0, y0, x1, y1], outline=COLORS["FP"], width=3)
        draw.text((x0 + 2, y0 + 2), f"FP {conf:.2f}", fill=COLORS["FP"], font=font)
    return img


def save_quadrant_grid(class_name: str, examples: dict[str, tuple], out_path: Path):
    """examples: quadrant -> (PIL.Image, filename) or None."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for quadrant, (r, c) in GRID_LAYOUT.items():
        ax = axes[r][c]
        entry = examples.get(quadrant)
        if entry is not None:
            img, filename = entry
            ax.imshow(img)
            ax.set_xlabel(filename, fontsize=8)
        else:
            ax.text(0.5, 0.5, "no example found", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(GRID_TITLES[quadrant])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f'"{class_name}" confusion matrix examples', fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="runs/detect/runs/license_plate/license_plate_ft_interleaved/weights/best.pt",
    )
    parser.add_argument("--data", default=str(DATASET_DIR / "dataset.yaml"))
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--classes", default="license_plate,car,person")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--output", default="runs/confusion_review")
    parser.add_argument(
        "--device", default="cpu", type=resolve_device,
        help="Passed to model.predict(device=...) - verified via device_check.resolve_device as part of "
        "argument parsing itself (type=resolve_device). Default 'cpu'.",
    )
    args = parser.parse_args()

    dataset_config = yaml.safe_load(Path(args.data).read_text())
    names: dict[int, str] = dataset_config["names"]
    name_to_id = {v: k for k, v in names.items()}
    target_classes = [c.strip() for c in args.classes.split(",")]
    for c in target_classes:
        if c not in name_to_id:
            raise SystemExit(f"Unknown class {c!r}. Available: {sorted(name_to_id.values())}")

    manifest = DATASET_DIR / f"{args.split}.txt"
    image_paths = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if line:
            p = Path(line)
            image_paths.append(p if p.is_absolute() else (DATASET_DIR / p).resolve())
    print(f"[review] {len(image_paths)} images in split={args.split}")

    model = YOLO(args.model)

    # found[class_name][quadrant] -> (PIL.Image, filename) once found, then skipped
    found: dict[str, dict[str, tuple]] = {c: {} for c in target_classes}

    def all_classes_complete() -> bool:
        return all(len(found[c]) == 4 for c in target_classes)

    def process_image(image_path: Path):
        img = Image.open(image_path)
        img_w, img_h = img.size
        gt_by_class = parse_yolo_labels(label_path_for_image(image_path), img_w, img_h)

        result = model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False, device=args.device)[0]
        pred_by_class: dict[int, list[tuple]] = {}
        if result.boxes is not None:
            for box_xyxy, cls_id, conf in zip(
                result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
            ):
                pred_by_class.setdefault(int(cls_id), []).append((tuple(box_xyxy), conf))

        plate_cid = name_to_id.get("license_plate")
        plate_boxes_conf = pred_by_class.get(plate_cid, []) if plate_cid is not None else []

        for class_name in target_classes:
            if len(found[class_name]) == 4:
                continue
            cid = name_to_id[class_name]
            gt_boxes = gt_by_class.get(cid, [])
            pred_boxes_conf = pred_by_class.get(cid, [])
            pred_boxes = [b for b, _ in pred_boxes_conf]
            # Show plate detections for spatial context on every non-plate
            # class's example images (TP/FP/FN/TN alike) - not part of this
            # class's own confusion accounting, just a visual reference.
            context_plates = plate_boxes_conf if class_name != "license_plate" else None

            matched_gt, matched_pred = match_boxes(gt_boxes, pred_boxes, args.iou_thresh)
            fn_idx = set(range(len(gt_boxes))) - matched_gt
            fp_idx = set(range(len(pred_boxes))) - matched_pred

            has_tp, has_fp, has_fn = bool(matched_gt), bool(fp_idx), bool(fn_idx)
            is_tn = not gt_boxes and not pred_boxes

            if not (has_tp or has_fp or has_fn or is_tn):
                continue

            annotated = None
            if has_tp or has_fp or has_fn:
                annotated = render_annotated(
                    image_path, gt_boxes, pred_boxes_conf, matched_gt, fp_idx, fn_idx, context_plates
                )

            if has_tp and "TP" not in found[class_name]:
                found[class_name]["TP"] = (annotated, image_path.name)
            if has_fp and "FP" not in found[class_name]:
                found[class_name]["FP"] = (annotated, image_path.name)
            if has_fn and "FN" not in found[class_name]:
                found[class_name]["FN"] = (annotated, image_path.name)
            if is_tn and "TN" not in found[class_name]:
                tn_img = img.convert("RGB")
                if context_plates:
                    draw_plate_context(ImageDraw.Draw(tn_img), context_plates, _load_font())
                found[class_name]["TN"] = (tn_img, image_path.name)

    for image_path in image_paths:
        if all_classes_complete():
            break
        process_image(image_path)

    if not all_classes_complete():
        # Every image in data/license_plates/ has a manually-labeled plate
        # (this repo's dataset was built plate-first), so license_plate can
        # never get a TN example from that split alone - pull in COCO128
        # (the same "non-plate" pool used for interleaved training, see
        # finetune_license_plate_interleaved.py): those images have zero
        # license_plate ground truth by construction, so any image where
        # the model also predicts no plate is a genuine TN (and one where
        # it does predict a plate is an interesting FP to see).
        print("[review] Some quadrants still missing after the primary split - pulling in COCO128 images")
        from ultralytics.data.utils import check_det_dataset

        from finetune_license_plate_interleaved import _list_images

        coco128_info = check_det_dataset("coco128.yaml")
        for image_path in _list_images(coco128_info["train"]):
            if all_classes_complete():
                break
            process_image(image_path)

    output_dir = Path(args.output)
    print(
        "[review] NOTE: ground truth for classes other than license_plate is "
        "machine pseudo-labeled (pseudo_label_plates.py) - FP/FN partly reflects "
        "pseudo-label noise, not only real model mistakes."
    )
    for class_name in target_classes:
        counts = {q: (q in found[class_name]) for q in ("TP", "FP", "FN", "TN")}
        out_path = output_dir / f"{class_name}_confusion_grid.png"
        save_quadrant_grid(class_name, found[class_name], out_path)
        print(f"[review] {class_name}: found={counts} -> {out_path}")


if __name__ == "__main__":
    main()
