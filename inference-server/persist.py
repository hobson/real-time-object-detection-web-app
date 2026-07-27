"""Best-effort persistence of submitted images + detections to the DB.

Shared by main.py's `/predict` and alpr.py's `/alpr/predict` + `/alpr/ws`.
Any failure here (DB down, disk full, etc.) is logged and swallowed —
persistence is a side channel for later curation, not part of the inference
contract, and must never turn a successful inference into a failed request.
"""
import hashlib
import io
import logging
import platform
import socket
from collections import Counter
from pathlib import Path

from PIL import Image
from PIL.ExifTags import GPSTAGS, IFD, TAGS

from orm import DetectionLabel, SubmittedImage
from db import SessionLocal

logger = logging.getLogger("persist")

STORAGE_DIR = Path(__file__).parent / "storage" / "images"
THUMBNAIL_DIR = Path(__file__).parent / "storage" / "thumbnails"
THUMBNAIL_MAX_SIZE = (128, 128)

# Static for the process lifetime - which machine ran inference, for
# capture_metadata's "server_host" key (see _build_capture_metadata). Useful
# once this server runs on more than one host, or under more than one
# execution provider.
_HOST_INFO = {"hostname": socket.gethostname(), "platform": platform.platform()}


def _clean_exif_value(value):
    """EXIF values come back as a grab-bag of PIL-specific types (raw bytes
    for MakerNote-style blobs, IFDRational for things like FocalLength) that
    aren't JSON-serializable as-is. Returns None for anything not worth
    keeping (raw bytes, empty strings)."""
    if isinstance(value, bytes):
        return None
    if isinstance(value, tuple):
        cleaned = [_clean_exif_value(v) for v in value]
        return cleaned if any(v is not None for v in cleaned) else None
    if isinstance(value, str):
        return value.strip("\x00 ") or None
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return float(value) if value.denominator else None
    return value


def _extract_exif(data: bytes) -> dict:
    """Best-effort EXIF extraction (camera make/model/focal length,
    orientation, GPS, capture timestamp) from a submitted JPEG. Returns {}
    for images with no EXIF - which is the common case for this app's own
    webcam capture path (a <canvas> frame from getUserMedia carries no
    camera metadata at all), but a real photo uploaded directly (e.g. via
    curl or the Swagger UI file picker) usually does."""
    try:
        exif = Image.open(io.BytesIO(data)).getexif()
    except Exception:
        return {}
    if not exif:
        return {}

    # ExifOffset/GPSInfo at the top level are just byte offsets pointing at
    # the sub-IFDs pulled out below, not real fields.
    _POINTER_TAGS = {"ExifOffset", "GPSInfo"}

    result = {}
    for tag_id, value in exif.items():
        tag = TAGS.get(tag_id, str(tag_id))
        if tag in _POINTER_TAGS:
            continue
        cleaned = _clean_exif_value(value)
        if cleaned is not None:
            result[tag] = cleaned

    for ifd_id in (IFD.Exif, IFD.GPSInfo):
        try:
            ifd = exif.get_ifd(ifd_id)
        except Exception:
            continue
        tagset = GPSTAGS if ifd_id == IFD.GPSInfo else TAGS
        sub = {}
        for tag_id, value in ifd.items():
            cleaned = _clean_exif_value(value)
            if cleaned is not None:
                sub[tagset.get(tag_id, str(tag_id))] = cleaned
        if sub:
            if ifd_id == IFD.GPSInfo:
                result["GPSInfo"] = sub
            else:
                result.update(sub)

    return result


def _detection_summary(detections: list[dict]) -> dict:
    classes = Counter(
        d.get("class_name") or d.get("class")
        for d in detections
        if d.get("class_name") or d.get("class")
    )
    return {"count": len(detections), "classes": dict(classes)}


def _build_capture_metadata(
    client_metadata: dict | None, image_bytes: bytes, detections: list[dict]
) -> dict:
    """capture_metadata used to be populated only from client-supplied GPS/
    orientation data (see request_parsing.py) - which no shipped client
    actually sends, so the column was always empty in practice. Fill it
    server-side instead with whatever we can determine ourselves: EXIF
    pulled from the image bytes, which host ran inference, and a summary of
    what was detected - so the field is never empty for a fresh submission,
    regardless of what (if anything) the caller attached."""
    metadata = dict(client_metadata or {})
    exif = _extract_exif(image_bytes)
    if exif:
        metadata["exif"] = exif
    metadata["server_host"] = _HOST_INFO
    metadata["detection_summary"] = _detection_summary(detections)
    return metadata


def _store_image(data: bytes, content_type: str | None) -> str:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    sha256 = hashlib.sha256(data).hexdigest()
    ext = {"image/jpeg": ".jpg", "image/png": ".png"}.get(content_type or "", ".jpg")
    path = STORAGE_DIR / f"{sha256}{ext}"
    if not path.exists():
        path.write_bytes(data)
    return sha256, str(path)


def _store_thumbnail(data: bytes, sha256: str) -> str | None:
    """Downscale the submitted image to fit THUMBNAIL_MAX_SIZE and save it
    as a JPEG for the admin list view. Returns None (not raised) on any
    decode failure - a missing thumbnail is not worth failing the request
    or the submission over, same best-effort contract as persist_submission
    itself."""
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    path = THUMBNAIL_DIR / f"{sha256}.jpg"
    if path.exists():
        return str(path)
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.thumbnail(THUMBNAIL_MAX_SIZE)
        image.save(path, format="JPEG", quality=80)
        return str(path)
    except Exception:
        logger.exception("Failed to generate thumbnail for %s", sha256)
        return None


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
) -> int | None:
    """Persist one submitted image plus its detections. `detections` items
    may include any of: class_id, class_name, box (normalized [x0,y0,x1,y1]
    or a (x_center,y_center,width,height) tuple already normalized),
    confidence, plate_text, ocr_confidence, region, region_confidence.

    `capture_metadata` is the opportunistic GPS/orientation/acceleration/
    camera-facing blob a multipart caller may have attached (see
    request_parsing.py) - merged (not replaced) with server-derived data
    added by `_build_capture_metadata`: EXIF pulled from the image bytes
    (camera make/model/focal length/GPS, when present), which host ran
    inference, and a summary of what was detected. `client_detections` is
    that same caller's own detection payload (e.g. it already ran
    in-browser YOLO) - same shape as `detections`, persisted as
    DetectionLabel rows with `source="client"` instead of the default
    `"server"`, so the two can be told apart later.

    Returns the new SubmittedImage row's id, or None if persistence itself
    failed (in which case there's no row for a caller to later attach a
    background-generated description to - see describe.py).
    """
    try:
        sha256, file_path = _store_image(image_bytes, content_type)
        thumbnail_path = _store_thumbnail(image_bytes, sha256)
        capture_metadata = _build_capture_metadata(capture_metadata, image_bytes, detections)
        session = SessionLocal()
        try:
            submitted = SubmittedImage(
                sha256=sha256,
                file_path=file_path,
                thumbnail_path=thumbnail_path,
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
            return submitted.id
        finally:
            session.close()
    except Exception:
        logger.exception("Failed to persist submission for endpoint %s", endpoint)
        return None


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
