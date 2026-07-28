"""Cross-model-agreement label confidence scoring (items 3+6 of the label-
quality investigation), combined with persisting item 2's "at least 2
models' machine labels" (item 2/task 15) - both need the same fresh
inference pass over every DB-loaded image, so this script does them
together rather than paying for that inference twice.

No "label quality" model exists or is buildable without a circular
ground-truth problem (see the design discussion this script's plan came
from) - instead, per-label confidence is a proxy: the fraction of
independent reference models whose own fresh detection over the same image
confirms that label (IoU>=0.5 + matching class), aggregated per-image as
the mean across its labels' scores.

Reference models (three, deliberately different lineages so they don't
share the same blind spots):
  1. --current-model: the COCO-capable checkpoint this run is evaluating/
     improving (default: round3's best.pt, but this must be re-pointed at
     each new round's own checkpoint - see --current-model-name and
     train_from_db.py's docstring for why staying in sync matters) -
     persisted as new DetectionLabel rows (label_source="machine") if not
     already present.
  2. The standalone license-plate detector (single-class, always outputs
     class 0 - remapped to LICENSE_PLATE_CLASS_ID for comparison/storage)
     - also persisted, satisfying item 2's "AND a model specifically
     trained to detect license plate objects."
  3. yolo11x.pt (ultralytics auto-download, item 6's "more accurate
     reference model") - used only to strengthen the agreement signal,
     not persisted as its own labels (it's an evaluator here, not a new
     labeling pass someone asked to keep).

Usage: python training/compute_label_confidence.py [--limit N] [--sources kitti,coco128,license_plates] \
    [--current-model path/to/best.pt] [--current-model-name some_round_name]
"""
import argparse
import sys
from pathlib import Path

import onnxruntime as ort
import yaml
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

REPO_ROOT = Path(__file__).parent.parent
INFERENCE_SERVER_DIR = REPO_ROOT / "inference-server"
PLATES_DIR = REPO_ROOT / "data" / "license_plates"

load_dotenv(INFERENCE_SERVER_DIR / ".env")
sys.path.insert(0, str(INFERENCE_SERVER_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from orm import DetectionLabel, SubmittedImage, engine_from_env  # noqa: E402
from persist import _detection_label  # noqa: E402
from postprocess import preprocess as _onnx_preprocess  # noqa: E402
from review_confusion_matrix import iou  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

LICENSE_PLATE_CLASS_ID = 80
NAMES: dict[int, str] = yaml.safe_load((PLATES_DIR / "dataset.yaml").read_text())["names"]
IOU_MATCH_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25

DEFAULT_CURRENT_CHECKPOINT = (
    REPO_ROOT / "runs" / "detect" / "runs" / "license_plate"
    / "license_plate_ft_reweight_round3" / "weights" / "best.pt"
)
DEFAULT_CURRENT_MODEL_NAME = "license_plate_ft_reweight_round3"
PLATE_CHECKPOINT = REPO_ROOT / "models" / "yolo-v9-t-384-license-plate-end2end.onnx"
PLATE_MODEL_NAME = "yolo-v9-t-384-license-plate-end2end"
PLATE_RESOLUTION = (384, 384)


def _to_xyxy(cx, cy, w, h) -> tuple[float, float, float, float]:
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _run_model(model: YOLO, image_path: str, imgsz) -> list[tuple[int, tuple]]:
    """Returns [(unified_class_id, (x0,y0,x1,y1) normalized), ...] for
    detections above CONF_THRESHOLD.

    device="cpu" is explicit, not incidental: on taco (ROCm-enabled torch),
    letting ultralytics auto-select the GPU segfaulted the process outright
    for this model/op combination on that iGPU (gfx1151) - this script runs
    once over the whole dataset and needs to not crash more than it needs
    GPU speed, so CPU is the safer default everywhere it runs."""
    result = model.predict(image_path, imgsz=imgsz, conf=CONF_THRESHOLD, verbose=False, device="cpu")[0]
    if result.boxes is None:
        return []
    w, h = result.orig_shape[1], result.orig_shape[0]
    detections = []
    for box_xyxy, cls_id in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
        x0, y0, x1, y1 = box_xyxy
        detections.append((int(cls_id), (x0 / w, y0 / h, x1 / w, y1 / h)))
    return detections


def _run_plate_model(ort_session: ort.InferenceSession, image_path: str) -> list[tuple[int, tuple]]:
    """This ONNX export has a static (non-dynamic) input shape and its own
    single-class scheme (always class 0 = license_plate) - ultralytics'
    YOLO() wrapper auto-letterboxes to a *rectangular* shape matching the
    source image's aspect ratio, which this fixed-shape model rejects with
    a dimension mismatch. Runs it the same way inference-server's own
    main.py does instead: onnxruntime directly + postprocess.py's
    preprocess() (a plain square resize), sidestepping ultralytics
    entirely for this one model."""
    input_name = ort_session.get_inputs()[0].name
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        tensor = _onnx_preprocess(img, PLATE_RESOLUTION)
        output = ort_session.run(None, {input_name: tensor})[0]
    detections = []
    for row in output:
        _, x0, y0, x1, y1, _cls_id, score = (float(v) for v in row)
        if score < CONF_THRESHOLD:
            continue
        w, h = PLATE_RESOLUTION
        detections.append((LICENSE_PLATE_CLASS_ID, (max(0.0, x0 / w), max(0.0, y0 / h), min(1.0, x1 / w), min(1.0, y1 / h))))
    return detections


def _persist_new_detections(
    session: Session, submitted: SubmittedImage, model_name: str, detections: list[tuple[int, tuple]]
) -> int:
    """Adds a DetectionLabel for each fresh detection that doesn't already
    match (IoU>=0.5, same class) an existing row from this same model_name
    on this image - keeps reruns from duplicating rows."""
    existing = [
        (d.class_id, (d.x_center - d.width / 2, d.y_center - d.height / 2, d.x_center + d.width / 2, d.y_center + d.height / 2))
        for d in submitted.detections
        if d.model_name == model_name
    ]
    added = 0
    for cls_id, box in detections:
        if any(cls_id == e_cls and iou(box, e_box) >= IOU_MATCH_THRESHOLD for e_cls, e_box in existing):
            continue
        det = {"class_id": cls_id, "class_name": NAMES.get(cls_id), "box": box}
        session.add(_detection_label(
            submitted.id, det, model_name=model_name, source="server", label_source="machine",
        ))
        added += 1
    return added


def _score_image(session: Session, submitted: SubmittedImage, models_and_detections) -> None:
    """models_and_detections: [(model_name, [(class_id, box), ...]), ...] -
    every reference model's fresh output for this image, used (regardless
    of whether that model's own output gets persisted) to score every
    DetectionLabel currently on this image."""
    session.flush()
    # Queried directly rather than via submitted.detections: that
    # relationship collection was loaded by the original query earlier in
    # main() and does NOT pick up rows _persist_new_detections has since
    # session.add()'ed - flush() sends the INSERTs but doesn't retroactively
    # append them to an already-loaded Python-side collection.
    labels = session.query(DetectionLabel).filter_by(submitted_image_id=submitted.id).all()
    scores = []
    for label in labels:
        label_box = _to_xyxy(label.x_center, label.y_center, label.width, label.height)
        agreeing = sum(
            1
            for _, detections in models_and_detections
            if any(cid == label.class_id and iou(label_box, box) >= IOU_MATCH_THRESHOLD for cid, box in detections)
        )
        label.cross_model_agreement = agreeing / len(models_and_detections)
        scores.append(label.cross_model_agreement)
    submitted.label_quality_score = sum(scores) / len(scores) if scores else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="kitti,coco128,license_plates")
    parser.add_argument("--limit", type=int, default=None, help="Cap total images scored (for smoke-testing).")
    parser.add_argument(
        "--current-model", default=str(DEFAULT_CURRENT_CHECKPOINT),
        help="The COCO-capable checkpoint currently being evaluated/improved (e.g. this round's "
        "training output) - re-run this script against each new round's checkpoint so train_from_db.py's "
        "oversampling reflects that round's actual mistakes, not a stale earlier one.",
    )
    parser.add_argument(
        "--current-model-name", default=DEFAULT_CURRENT_MODEL_NAME,
        help="model_name to persist this checkpoint's detections under - train_from_db.py's --current-model-name must match.",
    )
    args = parser.parse_args()
    source_tags = [s.strip() for s in args.sources.split(",")]

    # (model_path, imgsz, model_name, persist_as_labels) for the two models
    # ultralytics.YOLO() can load and run directly - the current checkpoint
    # (already in the unified 81-class space) and yolo11x.pt (standard COCO
    # ordering matching COCO's first 80 unified ids). The plate detector is
    # handled separately (_run_plate_model) - see its docstring for why.
    reference_models = [
        (args.current_model, 256, args.current_model_name, True),
        ("yolo11x.pt", 640, "yolo11x", False),
    ]

    print("[confidence] Loading reference models (yolo11x.pt auto-downloads on first use)...")
    models = [(YOLO(path), imgsz, name, persist) for path, imgsz, name, persist in reference_models]
    plate_session = ort.InferenceSession(str(PLATE_CHECKPOINT), providers=["CPUExecutionProvider"])

    engine = engine_from_env()
    with Session(engine) as session:
        images = (
            session.query(SubmittedImage)
            .join(SubmittedImage.tags)
            .filter(SubmittedImage.tags.any())
            .filter(SubmittedImage.endpoint == "training_import")
            .all()
        )
        # Filter to the requested source tags in Python (SQLAlchemy's
        # relationship .any() above already restricts to tagged rows;
        # exact tag-name matching is simpler done here than as SQL).
        images = [img for img in images if any(t.name in source_tags for t in img.tags)]
        if args.limit:
            images = images[: args.limit]
        print(f"[confidence] Scoring {len(images)} image(s)")

        for i, submitted in enumerate(images, 1):
            models_and_detections = []
            for model, imgsz, model_name, persist in models:
                detections = _run_model(model, submitted.file_path, imgsz)
                models_and_detections.append((model_name, detections))
                if persist:
                    n = _persist_new_detections(session, submitted, model_name, detections)
                    if n:
                        session.flush()

            plate_detections = _run_plate_model(plate_session, submitted.file_path)
            models_and_detections.append((PLATE_MODEL_NAME, plate_detections))
            n = _persist_new_detections(session, submitted, PLATE_MODEL_NAME, plate_detections)
            if n:
                session.flush()

            _score_image(session, submitted, models_and_detections)
            session.commit()
            if i % 25 == 0:
                print(f"[confidence] {i}/{len(images)} scored")

    print("[confidence] Done")


if __name__ == "__main__":
    main()
