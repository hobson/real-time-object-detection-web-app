"""Shared upload-size limit for main.py's `/predict` and alpr.py's
`/alpr/predict` + `/alpr/ws` — one JPEG frame is never anywhere near this
size, so it's just a guard against accidentally-huge/misbehaving clients.
"""
from fastapi import HTTPException

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5MB, plenty for a JPEG frame


def validate_body_size(body: bytes) -> None:
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")
