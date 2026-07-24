"""Per-model-family postprocessing, ported from components/models/Yolo.tsx
so server-side detection behavior matches the client-side WASM version.
"""
import numpy as np

from yolo_classes import YOLO_CLASSES

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.4
NUM_CLASSES = 80


def _iou(box_a, box_b):
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _apply_nms(detections, iou_threshold):
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = [True] * len(detections)
    for i in range(len(detections)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(detections)):
            if not keep[j]:
                continue
            if detections[i]["class_id"] != detections[j]["class_id"]:
                continue
            if _iou(detections[i]["box"], detections[j]["box"]) > iou_threshold:
                keep[j] = False
    return [d for d, k in zip(detections, keep) if k]


def _to_result(class_id, confidence, box, model_resolution):
    # Normalized to the model's input resolution (0..1) - the client scales
    # by its own canvas size, same as the existing client-side dx/dy math.
    width, height = model_resolution
    x0, y0, x1, y1 = (float(v) for v in box)
    return {
        "class": YOLO_CLASSES[class_id],
        "confidence": round(float(confidence), 4),
        "box": [
            max(0.0, x0 / width),
            max(0.0, y0 / height),
            min(1.0, x1 / width),
            min(1.0, y1 / height),
        ],
    }


def postprocess_yolov7(output: np.ndarray, model_resolution):
    # [det_num, 7]: batch_id, x0, y0, x1, y1, cls_id, score - already NMS'd
    results = []
    for row in output:
        _, x0, y0, x1, y1, cls_id, score = row
        results.append(
            _to_result(int(cls_id), score, (x0, y0, x1, y1), model_resolution)
        )
    return results


def postprocess_yolov10(output: np.ndarray, model_resolution):
    # [1, num_boxes, 6]: x0, y0, x1, y1, score, cls_id - pre-sorted by score
    # descending; stop at the first score below threshold (matches the
    # client's loop-with-break behavior).
    results = []
    for row in output[0]:
        x0, y0, x1, y1, score, cls_id = row
        if score < CONFIDENCE_THRESHOLD:
            break
        results.append(
            _to_result(int(cls_id), score, (x0, y0, x1, y1), model_resolution)
        )
    return results


def postprocess_yolov11_12(output: np.ndarray, model_resolution):
    # [1, 84, num_anchors]: 4 box coords (cx, cy, w, h) + 80 class scores
    # per anchor, raw (not NMS'd by the model) - decode, threshold, NMS.
    preds = output[0]  # [84, num_anchors]
    boxes_cxcywh = preds[:4, :].T  # [num_anchors, 4]
    class_scores = preds[4 : 4 + NUM_CLASSES, :].T  # [num_anchors, 80]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = confidences > CONFIDENCE_THRESHOLD
    boxes_cxcywh = boxes_cxcywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    cx, cy, w, h = (
        boxes_cxcywh[:, 0],
        boxes_cxcywh[:, 1],
        boxes_cxcywh[:, 2],
        boxes_cxcywh[:, 3],
    )
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    detections = [
        {
            "class_id": int(class_ids[i]),
            "confidence": float(confidences[i]),
            "box": [float(x0[i]), float(y0[i]), float(x1[i]), float(y1[i])],
        }
        for i in range(len(class_ids))
    ]
    detections = _apply_nms(detections, IOU_THRESHOLD)
    return [
        _to_result(d["class_id"], d["confidence"], d["box"], model_resolution)
        for d in detections
    ]


POSTPROCESS_MAP = {
    "yolo12n.onnx": postprocess_yolov11_12,
    "yolo11n.onnx": postprocess_yolov11_12,
    "yolov10n.onnx": postprocess_yolov10,
    "yolov7-tiny_256x256.onnx": postprocess_yolov7,
    "yolov7-tiny_320x320.onnx": postprocess_yolov7,
    "yolov7-tiny_640x640.onnx": postprocess_yolov7,
}
