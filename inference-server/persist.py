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

from PIL import Image as PILImage
from PIL.ExifTags import GPSTAGS, IFD, TAGS
from sqlalchemy.orm import Session

from orm import Annotation, Image, LabelSource
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


def _clean_tag_dict(items, tagset: dict) -> dict:
    """Shared by _extract_exif's top-level and per-IFD passes: resolve each
    tag id to its name and drop any value _clean_exif_value rejects."""
    result = {}
    for tag_id, value in items:
        cleaned = _clean_exif_value(value)
        if cleaned is not None:
            result[tagset.get(tag_id, str(tag_id))] = cleaned
    return result


def _extract_exif(data: bytes) -> dict:
    """Best-effort EXIF extraction (camera make/model/focal length,
    orientation, GPS, capture timestamp) from a submitted JPEG. Returns {}
    for images with no EXIF - which is the common case for this app's own
    webcam capture path (a <canvas> frame from getUserMedia carries no
    camera metadata at all), but a real photo uploaded directly (e.g. via
    curl or the Swagger UI file picker) usually does."""
    try:
        exif = PILImage.open(io.BytesIO(data)).getexif()
    except Exception:
        return {}
    if not exif:
        return {}

    result = _clean_tag_dict(exif.items(), TAGS)
    # ExifOffset/GPSInfo at the top level are just byte offsets pointing at
    # the sub-IFDs pulled out below, not real fields.
    result.pop("ExifOffset", None)
    result.pop("GPSInfo", None)

    for ifd_id in (IFD.Exif, IFD.GPSInfo):
        try:
            ifd = exif.get_ifd(ifd_id)
        except Exception:
            continue
        tagset = GPSTAGS if ifd_id == IFD.GPSInfo else TAGS
        sub = _clean_tag_dict(ifd.items(), tagset)
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


def _build_capture_metadata(client_metadata: dict | None, image_bytes: bytes) -> dict:
    """capture_metadata used to be populated only from client-supplied GPS/
    orientation data (see request_parsing.py) - which no shipped client
    actually sends, so the column was always empty in practice. Fill it
    server-side instead with whatever we can determine ourselves: EXIF
    pulled from the image bytes and which host ran inference - so the field
    is never empty for a fresh submission, regardless of what (if anything)
    the caller attached. Kept separate from detection_metadata (see
    _detection_summary) - one is about the capture device/environment, the
    other about what the model found; conflating them made either one
    harder to query."""
    metadata = dict(client_metadata or {})
    exif = _extract_exif(image_bytes)
    if exif:
        metadata["exif"] = exif
    metadata["server_host"] = _HOST_INFO
    return metadata


def _store_image(data: bytes, content_type: str | None) -> str:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    sha256 = hashlib.sha256(data).hexdigest()
    ext = {"image/jpeg": ".jpg", "image/png": ".png"}.get(content_type or "", ".jpg")
    path = STORAGE_DIR / f"{sha256}{ext}"
    if not path.exists():
        path.write_bytes(data)
    return sha256, str(path)


def _store_thumbnail(data: bytes, sha256: str, *, image: PILImage.Image | None = None) -> str | None:
    """Downscale the submitted image to fit THUMBNAIL_MAX_SIZE and save it
    as a JPEG for the admin list view. Returns None (not raised) on any
    decode failure - a missing thumbnail is not worth failing the request
    or the submission over, same best-effort contract as persist_submission
    itself.

    `image`, if given, is a caller-decoded Image reused instead of
    re-decoding `data` from scratch (main.py's /predict already decodes the
    same bytes for inference before persisting) - copied first since
    `.thumbnail()` resizes in place and callers still need their own copy
    afterwards."""
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    path = THUMBNAIL_DIR / f"{sha256}.jpg"
    if path.exists():
        return str(path)
    try:
        thumb = image.copy() if image is not None else PILImage.open(io.BytesIO(data)).convert("RGB")
        thumb.thumbnail(THUMBNAIL_MAX_SIZE)
        thumb.save(path, format="JPEG", quality=80)
        return str(path)
    except Exception:
        logger.exception("Failed to generate thumbnail for %s", sha256)
        return None


# ---------------------------------------------------------------------------
# LabelSource resolution - get-or-create so repeated calls for the same
# model/dataset identity reuse one row (see orm.py's LabelSource docstring).
# Shared by this module's own server-side persistence below and by
# training/db_load_training_images.py + training/compute_label_confidence.py
# (imported directly from here, same as they already import _annotation
# below) so there is exactly one place that knows how to resolve a
# LabelSource identity.
# ---------------------------------------------------------------------------

def get_or_create(session: Session, model_cls, defaults: dict | None = None, **filters):
    """Generic get-or-create: query `model_cls` by `filters`, or construct
    and flush a new row using `filters` plus `defaults` (extra fields only
    set on creation, not matched against - e.g. a Dataset's free-text
    `description`, which shouldn't be part of its identity lookup).
    Not module-private: also used by training/db_load_training_images.py's
    tag/dataset lookups, which do the identical query-then-insert-if-missing
    dance, so this pattern exists in exactly one place."""
    existing = session.query(model_cls).filter_by(**filters).one_or_none()
    if existing:
        return existing
    row = model_cls(**filters, **(defaults or {}))
    session.add(row)
    session.flush()
    return row


def get_or_create_model_label_source(
    session: Session,
    model_name: str | None,
    *,
    weights_hash: str | None = None,
    major_version: int | None = None,
    minor_version: int | None = None,
) -> LabelSource:
    """`model_name=None` (a server-computed detection with no reported model,
    which shouldn't happen in practice given main.py/alpr.py always pass one,
    but is guarded against here rather than raising) resolves to a single
    shared "unknown" identity rather than leaving label_source_id null -
    every Annotation must be attributable to *some* source."""
    return get_or_create(
        session, LabelSource, source_type="model", model_name=model_name or "unknown",
        weights_hash=weights_hash, major_version=major_version, minor_version=minor_version,
    )


def get_or_create_dataset_label_source(session: Session, dataset_id: int) -> LabelSource:
    return get_or_create(session, LabelSource, source_type="dataset", dataset_id=dataset_id)


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
    image: PILImage.Image | None = None,
) -> int | None:
    """Persist one submitted image plus its detections. `detections` items
    may include any of: class_id, class_name, box (normalized [x0,y0,x1,y1]
    or a (x_center,y_center,width,height) tuple already normalized),
    confidence, plate_text, ocr_confidence, region, region_confidence,
    model_name (overrides this call's own `model_name` for that one
    detection - e.g. ALPR's OCR stage vs. its detector stage).

    `capture_metadata` is about the capture device/environment: the
    opportunistic GPS/orientation/acceleration/camera-facing blob a
    multipart caller may have attached (see request_parsing.py), merged
    (not replaced) with server-derived data added by
    `_build_capture_metadata` - EXIF pulled from the image bytes (camera
    make/model/focal length/GPS, when present) and which host ran
    inference. `detection_metadata` is the separate, model-output side of
    things - a summary of what was detected (see `_detection_summary`) -
    kept in its own column so querying/indexing one doesn't drag in the
    other. `client_detections` is that same caller's own detection payload
    (e.g. it already ran in-browser YOLO) - same shape as `detections`,
    persisted as Annotation rows with `source="client"` instead of the
    default `"server"`, so the two can be told apart later.

    Returns the new Image row's id, or None if persistence itself failed (in
    which case there's no row for a caller to later attach a
    background-generated description to - see describe.py).

    `image` lets a caller that already decoded the bytes with PIL (main.py's
    /predict does, for inference) pass that Image along instead of paying
    for a second full JPEG decode inside _store_thumbnail. Every persisted
    row here defaults to `training_status="unreviewed"` (the Image column
    default) - this is live, unvetted production traffic; promoting one into
    the training corpus is a deliberate curator action in Flask-Admin, not
    something this path ever does itself.
    """
    try:
        sha256, file_path = _store_image(image_bytes, content_type)
        thumbnail_path = _store_thumbnail(image_bytes, sha256, image=image)
        capture_metadata = _build_capture_metadata(capture_metadata, image_bytes)
        detection_metadata = _detection_summary(detections)
        session = SessionLocal()
        try:
            submitted = Image(
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
                detection_metadata=detection_metadata,
            )
            session.add(submitted)
            session.flush()

            # Resolve one LabelSource per unique model name across this
            # whole batch, not once per detection - a single frame can carry
            # many detections from the same model, and each resolution is a
            # DB round trip (get-or-create).
            all_dets = [*detections, *(client_detections or [])]
            names = {det.get("model_name") or model_name for det in all_dets}
            label_source_ids = {name: get_or_create_model_label_source(session, name).id for name in names}

            for det in detections:
                label_source_id = label_source_ids[det.get("model_name") or model_name]
                session.add(_annotation(submitted.id, det, label_source_id, source="server"))
            for det in client_detections or []:
                label_source_id = label_source_ids[det.get("model_name") or model_name]
                session.add(_annotation(submitted.id, det, label_source_id, source="client"))
            session.commit()
            return submitted.id
        finally:
            session.close()
    except Exception:
        logger.exception("Failed to persist submission for endpoint %s", endpoint)
        return None


def _annotation(image_id: int, det: dict, label_source_id: int, *, source: str) -> Annotation:
    """Build an (unsaved) Annotation row from a detection dict plus an
    already-resolved `label_source_id` (see get_or_create_*_label_source
    above - callers resolve the identity, this just assembles the row).
    `source` ("server"/"client") is who computed this at request time,
    orthogonal to label_source_id's "how did this label come to exist" -
    see orm.py's Annotation.source docstring."""
    x_center, y_center, box_width, box_height = _to_center_wh(det["box"])
    return Annotation(
        image_id=image_id,
        label_source_id=label_source_id,
        class_id=det.get("class_id"),
        class_name=det.get("class_name") or det.get("class"),
        x_center=x_center,
        y_center=y_center,
        width=box_width,
        height=box_height,
        confidence=det.get("confidence"),
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
