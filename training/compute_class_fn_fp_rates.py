"""Aggregate per-class TP/FP/FN counts across a full validation split, and
compute a class_weights tensor from them via an F-beta score - the
accuracy-driven counterpart to train_resume_all_categories.py's frequency-
based 'balanced' weights. Targets the failure mode actually measured after
a training round (classes the model performs poorly on, weighted toward
"missed" over "false-alarmed" since a missed pedestrian/vehicle matters
more than a spurious box for the self-driving use case this model targets),
rather than proxying for that via how many training examples each class has.

  Fbeta = (1+b^2)*TP / ((1+b^2)*TP + b^2*FN + FP), b = --beta (default 2.0)
  weight = 1 + (max_weight - 1) * (1 - Fbeta), clamped to [min_weight, max_weight]

beta > 1 counts recall (avoiding FN) more heavily than precision (avoiding
FP) in Fbeta - beta=1 is plain F1 (FN and FP weighted equally), beta=2
(the default) weights recall ~4x precision, matching this project's
priority on not missing objects over not over-triggering.

This replaces an earlier fn_rate/fp_rate *ratio* formula, dropped after
round 1 of iterative_reweight_train.py's training loop showed two concrete
problems with it: (a) a class with zero true positives (e.g. "boat": tp=0
fp=5 fn=1) computed fn_rate=1.0, fp_rate=1.0 -> ratio=1.0, i.e. *neutral* -
the ratio is blind to absolute performance, only sensitive to the FN/FP
*balance*, so a class doing uniformly badly at both looked identical to one
doing fine at both; (b) small samples produced unstably extreme ratios
(one class hit the weight ceiling off 3 total instances). Fbeta fixes (a)
directly (a class with zero TP scores Fbeta=0, correctly flagged as
maximally bad) and is naturally bounded to [0,1], avoiding the ratio's
divide-by-near-zero blowups that caused (b).

Two classes of degenerate cases still need explicit handling, since Fbeta's
precision/recall components are each undefined (0/0) when their denominator
is zero:
  - No evidence at all (tp=fp=fn=0): weight=1.0 (nothing to conclude).
  - Pure hallucination (tp=0, fn=0, fp>0 - the class never appears as
    ground truth anywhere in the val split, but the model predicts it
    anyway): recall is undefined (no GT to have missed) while precision is
    a real 0 - plugged naively into the Fbeta formula this reads as
    "perfect recall, zero precision" and would incorrectly reward it with
    a high weight. Explicitly down-weighted to min_weight instead - these
    are genuine hallucinations (seen for "elephant"/"giraffe" against
    KITTI/plate images in round 1's data) that should be discouraged, not
    reinforced.

Box-level TP/FP/FN counting (per class, per image, via IoU matching) reuses
review_confusion_matrix.py's matching logic directly rather than
reimplementing it - that script already does exactly this per-image, just
keeps only the first example per quadrant instead of aggregate counts.

Usage: python training/compute_class_fn_fp_rates.py \
    --model runs/detect/runs/license_plate/license_plate_ft_reweight_round1/weights/best.pt \
    --data data/external_datasets/all_categories_dataset.generated.yaml \
    --split val --beta 2.0 --out training/class_weights_round1.pt
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from ultralytics import YOLO

from finetune_license_plate_interleaved import _list_images
from review_confusion_matrix import label_path_for_image, match_boxes, parse_yolo_labels

EPS = 1e-3


def aggregate_counts(
    model: YOLO, image_paths: list[Path], num_classes: int, imgsz: int, conf: float, iou_thresh: float
) -> tuple[list[int], list[int], list[int]]:
    """Returns (tp, fp, fn) - one count per class, summed across every image
    in image_paths. Only classes that appear as ground truth and/or a
    prediction in a given image are touched for that image (a class absent
    from both contributes nothing to any of its three counts there, same as
    review_confusion_matrix.py's TN handling, but we don't track TN here -
    it isn't used by the fn_rate/fp_rate formulas above).
    """
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for image_path in image_paths:
        img = Image.open(image_path)
        img_w, img_h = img.size
        gt_by_class = parse_yolo_labels(label_path_for_image(image_path), img_w, img_h)

        result = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)[0]
        pred_by_class: dict[int, list[tuple]] = {}
        if result.boxes is not None:
            for box_xyxy, cls_id in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
                pred_by_class.setdefault(int(cls_id), []).append(tuple(box_xyxy))

        for cid in set(gt_by_class) | set(pred_by_class):
            gt_boxes = gt_by_class.get(cid, [])
            pred_boxes = pred_by_class.get(cid, [])
            matched_gt, matched_pred = match_boxes(gt_boxes, pred_boxes, iou_thresh)
            tp[cid] += len(matched_gt)
            fn[cid] += len(gt_boxes) - len(matched_gt)
            fp[cid] += len(pred_boxes) - len(matched_pred)

    return tp, fp, fn


def compute_weights(
    tp: list[int], fp: list[int], fn: list[int], beta: float, min_weight: float, max_weight: float
) -> tuple[torch.Tensor, list[dict]]:
    beta_sq = beta * beta
    weights = []
    per_class = []
    for c in range(len(tp)):
        recall_denom = tp[c] + fn[c]
        precision_denom = tp[c] + fp[c]

        if recall_denom == 0 and precision_denom == 0:
            f_beta = None  # no GT or predictions at all - no signal either way
            weight = 1.0
        elif recall_denom == 0 and fp[c] > 0:
            # Pure hallucination: never appears as GT anywhere in this split,
            # but the model predicts it - recall is undefined (nothing to
            # have missed), so naively plugging in "perfect recall" would
            # reward this with a high weight instead of discouraging it.
            f_beta = 0.0
            weight = min_weight
        else:
            precision = tp[c] / precision_denom if precision_denom > 0 else 1.0
            recall = tp[c] / recall_denom if recall_denom > 0 else 1.0
            denom = beta_sq * precision + recall
            f_beta = (1 + beta_sq) * precision * recall / denom if denom > 0 else 0.0
            weight = 1 + (max_weight - 1) * (1 - f_beta)
        weight = min(max(weight, min_weight), max_weight)

        weights.append(weight)
        per_class.append(
            {
                "tp": tp[c], "fp": fp[c], "fn": fn[c],
                "f_beta": round(f_beta, 4) if f_beta is not None else None,
                "weight": round(weight, 4),
            }
        )
    return torch.tensor(weights, dtype=torch.float32), per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Checkpoint to evaluate (.pt).")
    parser.add_argument("--data", required=True, help="dataset yaml with 'names' and a 'val'/'--split' key.")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--beta", type=float, default=2.0, help="F-beta's beta - >1 weights recall (avoiding misses) over precision (avoiding false alarms); 1.0 = plain F1.")
    parser.add_argument("--min-weight", type=float, default=0.1)
    parser.add_argument("--max-weight", type=float, default=10.0)
    parser.add_argument("--out", required=True, help="Where to save the class_weights tensor (.pt).")
    parser.add_argument("--report-out", default=None, help="Optional per-class TP/FP/FN/rate/weight JSON, for auditing (defaults to --out with .json instead of .pt).")
    args = parser.parse_args()

    dataset_config = yaml.safe_load(Path(args.data).read_text())
    names: dict[int, str] = dataset_config["names"]
    num_classes = len(names)

    image_paths = _list_images(dataset_config[args.split])
    print(f"[compute-weights] {len(image_paths)} images in split={args.split}, {num_classes} classes")

    model = YOLO(args.model)
    tp, fp, fn = aggregate_counts(model, image_paths, num_classes, args.imgsz, args.conf, args.iou_thresh)
    weights, per_class = compute_weights(tp, fp, fn, args.beta, args.min_weight, args.max_weight)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out_path)

    report_path = Path(args.report_out) if args.report_out else out_path.with_suffix(".json")
    report = {names[c]: per_class[c] for c in range(num_classes)}
    report_path.write_text(json.dumps(report, indent=2))

    worst = sorted(range(num_classes), key=lambda c: -weights[c])[:10]
    print(f"[compute-weights] Top 10 up-weighted classes (lowest F{args.beta:g}, excluding no-evidence/pure-hallucination):")
    for c in worst:
        print(f"  {names[c]:20s} tp={tp[c]:5d} fp={fp[c]:5d} fn={fn[c]:5d} weight={weights[c]:.3f}")
    print(f"[compute-weights] Saved weights -> {out_path}")
    print(f"[compute-weights] Saved per-class report -> {report_path}")


if __name__ == "__main__":
    main()
