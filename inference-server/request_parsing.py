"""Shared image+metadata request parsing for main.py's `/predict` and
alpr.py's `/alpr/predict`.

Both endpoints accept two request shapes:
  - raw image bytes as the whole body (`Content-Type: image/jpeg`/`image/
    png`) - the original contract, unchanged, still the simplest way to
    call either endpoint (see each router's docstring for a curl example).
  - `multipart/form-data` with an `image` file part and an optional
    `metadata` JSON string part, for clients that also have GPS/IMU/camera-
    facing data or their own client-side YOLO detections to attach (see
    docs/user-manual.md's "Attaching capture metadata" section).

FastAPI's `File`/`Form` declarations can't express "this OR a raw body,
depending on Content-Type" - declaring either forces Starlette to parse
every request as multipart, which raises on a plain raw-JPEG POST. Hence
the manual `Request.headers` content-type check here instead.
"""
import json

from fastapi import HTTPException, Request


async def parse_image_and_metadata(request: Request) -> tuple[bytes, dict | None, str | None]:
    """Returns (image_bytes, metadata, image_content_type). For a raw body,
    image_content_type is just the request's own Content-Type header; for
    multipart, it's the `image` part's own Content-Type (not the outer
    `multipart/form-data; boundary=...`), so callers that persist/branch on
    it (e.g. _store_image's extension guess) see the real image type either
    way.
    """
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("multipart/form-data"):
        return await request.body(), None, content_type or None

    form = await request.form()
    image_field = form.get("image")
    if image_field is None or not hasattr(image_field, "read"):
        raise HTTPException(
            status_code=400, detail="multipart request missing 'image' file field"
        )
    image_bytes = await image_field.read()
    image_content_type = getattr(image_field, "content_type", None) or "image/jpeg"

    metadata = None
    raw_metadata = form.get("metadata")
    if raw_metadata:
        try:
            metadata = json.loads(raw_metadata)
        except (json.JSONDecodeError, TypeError) as e:
            raise HTTPException(
                status_code=400, detail="'metadata' field must be valid JSON"
            ) from e
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="'metadata' must be a JSON object")

    return image_bytes, metadata, image_content_type
