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
from db import get_session

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
) -> None:
    """Persist one submitted image plus its detections. `detections` items
    may include any of: class_id, class_name, box (normalized [x0,y0,x1,y1]
    or a (x_center,y_center,width,height) tuple already normalized),
    confidence, plate_text, ocr_confidence, region, region_confidence.
    """
    try:
        sha256, file_path = _store_image(image_bytes, content_type)
        session = get_session()
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
            )
            session.add(submitted)
            session.flush()

            for det in detections:
                x_center, y_center, box_width, box_height = _to_center_wh(det["box"])
                session.add(
                    DetectionLabel(
                        submitted_image_id=submitted.id,
                        class_id=det.get("class_id"),
                        class_name=det.get("class_name"),
                        x_center=x_center,
                        y_center=y_center,
                        width=box_width,
                        height=box_height,
                        confidence=det.get("confidence"),
                        model_name=model_name,
                        plate_text=det.get("plate_text"),
                        ocr_confidence=det.get("ocr_confidence"),
                        region=det.get("region"),
                        region_confidence=det.get("region_confidence"),
                    )
                )
            session.commit()
        finally:
            session.close()
    except Exception:
        logger.exception("Failed to persist submission for endpoint %s", endpoint)


def _to_center_wh(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert a normalized [x0, y0, x1, y1] box to YOLO center/width/height."""
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
