"""Per-model-family postprocessing, ported from components/models/Yolo.tsx
so server-side detection behavior matches the client-side WASM version.
"""
import numpy as np

from yolo_classes import YOLO_CLASSES

CONFIDENCE_THRESHOLD = 0.25


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


POSTPROCESS_MAP = {
    # yolo11n/yolo12n.onnx are exported with nms=True baked in (see
    # convert_pt_to_onnx/), so their output is already [1,300,6] pre-sorted
    # detections - same format as yolov10.
    "yolo12n.onnx": postprocess_yolov10,
    "yolo11n.onnx": postprocess_yolov10,
    "yolov10n.onnx": postprocess_yolov10,
    "yolov7-tiny_256x256.onnx": postprocess_yolov7,
    "yolov7-tiny_320x320.onnx": postprocess_yolov7,
    "yolov7-tiny_640x640.onnx": postprocess_yolov7,
}
