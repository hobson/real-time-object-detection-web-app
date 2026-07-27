"""Best-effort persistence of submitted images + detections to the DB.

Shared by main.py's `/predict` and alpr.py's `/alpr/predict` + `/alpr/ws`.
Any failure here (DB down, disk full, etc.) is logged and swallowed —
persistence is a side channel for later curation, not part of the inference
contract, and must never turn a successful inference into a failed request.
"""
import hashlib
import logging
from pathlib import Path

from orm import DetectionLabel, SubmittedImage
from db import SessionLocal

logger = logging.getLogger("persist")

STORAGE_DIR = Path(__file__).parent / "storage" / "images"


def _store_image(data: bytes, content_type: str | None) -> str:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    sha256 = hashlib.sha256(data).hexdigest()
    ext = {"image/jpeg": ".jpg", "image/png": ".png"}.get(content_type or "", ".jpg")
    path = STORAGE_DIR / f"{sha256}{ext}"
    if not path.exists():
        path.write_bytes(data)
    return sha256, str(path)


def persist_submission(
    *,
    image_bytes: bytes,
    endpoint: str,
    width: int | None,
    height: int | None,
    content_type: str | None = None,
    model_name: str | None = None,
    client_ip: str | None = None,
    inference_time_ms: float | None = None,
    status_code: int | None = None,
    detections: list[dict],
    capture_metadata: dict | None = None,
    client_detections: list[dict] | None = None,
) -> None:
    """Persist one submitted image plus its detections. `detections` items
    may include any of: class_id, class_name, box (normalized [x0,y0,x1,y1]
    or a (x_center,y_center,width,height) tuple already normalized),
    confidence, plate_text, ocr_confidence, region, region_confidence.

    `capture_metadata` is the opportunistic GPS/orientation/acceleration/
    camera-facing blob a multipart caller may have attached (see
    request_parsing.py) - stored as-is, `None` for the plain-raw-body
    request shape. `client_detections` is that same caller's own detection
    payload (e.g. it already ran in-browser YOLO) - same shape as
    `detections`, persisted as DetectionLabel rows with `source="client"`
    instead of the default `"server"`, so the two can be told apart later.
    """
    try:
        sha256, file_path = _store_image(image_bytes, content_type)
        session = SessionLocal()
        try:
            submitted = SubmittedImage(
                sha256=sha256,
                file_path=file_path,
                width=width,
                height=height,
                content_type=content_type,
                endpoint=endpoint,
                model_name=model_name,
                client_ip=client_ip,
                inference_time_ms=inference_time_ms,
                status_code=status_code,
                capture_metadata=capture_metadata,
            )
            session.add(submitted)
            session.flush()

            for det in detections:
                session.add(_detection_label(submitted.id, det, model_name, source="server"))
            for det in client_detections or []:
                session.add(_detection_label(submitted.id, det, model_name, source="client"))
            session.commit()
        finally:
            session.close()
    except Exception:
        logger.exception("Failed to persist submission for endpoint %s", endpoint)


def _detection_label(submitted_image_id: int, det: dict, model_name: str | None, *, source: str) -> DetectionLabel:
    x_center, y_center, box_width, box_height = _to_center_wh(det["box"])
    return DetectionLabel(
        submitted_image_id=submitted_image_id,
        class_id=det.get("class_id"),
        class_name=det.get("class_name") or det.get("class"),
        x_center=x_center,
        y_center=y_center,
        width=box_width,
        height=box_height,
        confidence=det.get("confidence"),
        model_name=det.get("model_name") or model_name,
        plate_text=det.get("plate_text"),
        ocr_confidence=det.get("ocr_confidence"),
        region=det.get("region"),
        region_confidence=det.get("region_confidence"),
        source=source,
    )


def _to_center_wh(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert a normalized [x0, y0, x1, y1] box to YOLO center/width/height."""
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
