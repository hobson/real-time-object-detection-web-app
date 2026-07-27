"""License plate detection + OCR endpoint, backed by `fast-alpr`.

Transport: WebSocket, not per-frame HTTP POST (like `/predict` in main.py)
and not WebRTC. Reasoning:

- WebRTC is built for continuous, high-fps video where its whole value
  proposition - temporal compression (only encode deltas between frames),
  jitter buffering, adaptive bitrate - pays off. At 1 fps there is no
  "previous frame" worth predicting from: every frame is effectively a
  keyframe, so a video codec buys nothing over a plain JPEG still and only
  adds cost (SDP/ICE negotiation, a TURN relay in most NATed deployments,
  a decode pipeline on the server). It's the wrong tool for "one still
  image per second".
- Plain HTTP POST per frame (what /predict does for the COCO models) is
  fine at 1 fps too - the per-request overhead (TCP/TLS handshake or
  HTTP/1.1 keep-alive + headers, ~0.5-1KB) is trivial next to a ~30-80KB
  JPEG at that rate. But it can't push results back without the client
  polling, and it pays connection setup on every request unless keep-alive
  is correctly reused end-to-end (not guaranteed through the Tailscale
  Funnel path this deploys behind - see CLAUDE.md's inference-server
  deployment note).
- A single persistent WebSocket amortizes the handshake/TLS cost across
  the whole session (one setup instead of one per frame), has ~2-14 bytes
  of framing overhead per message instead of a full HTTP request line +
  headers, and lets the server push the JSON result back the moment it's
  ready - a natural fit for "client streams frames in, server streams
  results back" without polling. At 1 fps the absolute bytes saved vs.
  HTTP are small, but it's the closest match to the actual shape of the
  problem (a long-lived bidirectional low-rate stream) with no added
  complexity over HTTP.

Pacing: the client is expected to send at ~1 fps: this handler processes
messages strictly in order as they arrive and does not buffer/skip frames
itself, so a client sending faster than the server can process will simply
queue up (backpressure via TCP), which is why the client-side helper
(`components/models/AlprServer.tsx`) paces itself to a 1s minimum interval
between sends rather than relying on the server to drop frames.
"""
import logging
import os
import statistics
import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from open_image_models.detection.core.hub import PlateDetectorModel
from fast_plate_ocr.inference.hub import OcrModel

from fast_alpr import ALPR
from limits import MAX_UPLOAD_BYTES, validate_body_size
from persist import persist_submission
from request_parsing import parse_image_and_metadata

logger = logging.getLogger("alpr")

router = APIRouter(prefix="/alpr", tags=["alpr"])

DEFAULT_DETECTOR_MODEL: PlateDetectorModel = os.environ.get(
    "ALPR_DETECTOR_MODEL", "yolo-v9-t-384-license-plate-end2end"
)
DEFAULT_OCR_MODEL: OcrModel = os.environ.get("ALPR_OCR_MODEL", "cct-xs-v2-global-model")

_alpr: ALPR | None = None


def get_alpr() -> ALPR:
    # No lock needed - same reasoning as main.py's get_session().
    global _alpr
    if _alpr is None:
        logger.info(
            "Loading ALPR models: detector=%s ocr=%s", DEFAULT_DETECTOR_MODEL, DEFAULT_OCR_MODEL
        )
        _alpr = ALPR(
            detector_model=DEFAULT_DETECTOR_MODEL,
            ocr_model=DEFAULT_OCR_MODEL,
            detector_providers=["CPUExecutionProvider"],
            ocr_providers=["CPUExecutionProvider"],
        )
    return _alpr


def _decode_jpeg(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode JPEG frame")
    return frame


def _run_alpr(frame: np.ndarray) -> dict:
    height, width = frame.shape[:2]
    alpr = get_alpr()

    start = time.time()
    results = alpr.predict(frame)
    inference_ms = (time.time() - start) * 1000

    detections = []
    for result in results:
        bbox = result.detection.bounding_box
        ocr = result.ocr
        detections.append(
            {
                "box": [bbox.x1 / width, bbox.y1 / height, bbox.x2 / width, bbox.y2 / height],
                "detectionConfidence": result.detection.confidence,
                "plate": ocr.text if ocr else None,
                "ocrConfidence": (
                    statistics.mean(ocr.confidence)
                    if ocr and isinstance(ocr.confidence, list)
                    else (ocr.confidence if ocr else None)
                ),
                "region": ocr.region if ocr else None,
                "regionConfidence": ocr.region_confidence if ocr else None,
            }
        )

    return {"inferenceTimeMs": round(inference_ms, 1), "detections": detections}


def _persist_alpr(
    *,
    frame: np.ndarray,
    jpeg_bytes: bytes,
    endpoint: str,
    client_ip: str | None,
    result: dict,
    capture_metadata: dict | None = None,
    client_detections: list[dict] | None = None,
) -> None:
    height, width = frame.shape[:2]
    persist_submission(
        image_bytes=jpeg_bytes,
        endpoint=endpoint,
        width=width,
        height=height,
        content_type="image/jpeg",
        model_name=DEFAULT_DETECTOR_MODEL,
        client_ip=client_ip,
        inference_time_ms=result["inferenceTimeMs"],
        status_code=200,
        detections=[
            {
                "class_name": "license_plate",
                "confidence": d["detectionConfidence"],
                "box": d["box"],
                "plate_text": d["plate"],
                "ocr_confidence": d["ocrConfidence"],
                "region": d["region"],
                "region_confidence": d["regionConfidence"],
            }
            for d in result["detections"]
        ],
        capture_metadata=capture_metadata,
        client_detections=client_detections,
    )


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict")
async def predict(request: Request):
    """Detect and OCR license plates in a single image.

    **Request body** - either:
    - raw JPEG bytes (`Content-Type: image/jpeg`), the original contract:

      ```bash
      curl -X POST "https://taco.tail9f615d.ts.net:10000/infer/alpr/predict" \\
        --data-binary @plate.jpg -H "Content-Type: image/jpeg"
      ```

      (The "Try it out" button below sends the same request — pick a file
      and hit Execute — but the endpoint doesn't accept a `curl -d ''`
      with no body, since a JPEG is required.)
    - `multipart/form-data` with an `image` file part and an optional
      `metadata` JSON string part (GPS/orientation/acceleration/camera-
      facing, and/or a `client_detections` array) - see
      docs/user-manual.md and main.py's `/predict` docstring, which
      documents the same shape.

    The image can be a close-up of a single plate or a full scene (e.g. a
    whole vehicle or a street photo); the detector locates every
    plate-shaped region in the frame first, then runs OCR on each one
    independently, so multiple plates in one image each get their own
    entry in `detections`.

    **Response**: `{"inferenceTimeMs": float, "detections": [...]}`. Each
    detection has `box` (`[x0, y0, x1, y1]`, normalized 0-1 against the
    image's own width/height), `detectionConfidence`, and the OCR result:
    `plate` (the recognized text), `ocrConfidence`, `region` (best-effort
    country/region guess), `regionConfidence`. The OCR fields are `null`
    when a plate box was found but the text couldn't be read confidently —
    that's a normal outcome for blurry, angled, or partially occluded
    plates, not an error.
    """
    body, metadata, _content_type = await parse_image_and_metadata(request)
    validate_body_size(body)
    client_detections = metadata.pop("client_detections", None) if metadata else None
    try:
        frame = _decode_jpeg(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    result = _run_alpr(frame)
    _persist_alpr(
        frame=frame,
        jpeg_bytes=body,
        endpoint="/alpr/predict",
        client_ip=request.client.host if request.client else None,
        result=result,
        capture_metadata=metadata,
        client_detections=client_detections,
    )
    return JSONResponse(result)


@router.websocket("/ws")
async def alpr_stream(websocket: WebSocket):
    """Persistent stream: client sends one binary JPEG frame per message,
    server replies with one JSON detections message per frame, in order.
    """
    await websocket.accept()
    try:
        get_alpr()  # load models before the first frame arrives
    except Exception as e:  # model load failure - fail the connection clearly
        await websocket.close(code=1011, reason=f"Model load failed: {e}")
        return

    try:
        while True:
            data = await websocket.receive_bytes()
            if len(data) > MAX_UPLOAD_BYTES:
                await websocket.send_json({"error": "Frame too large"})
                continue
            try:
                frame = _decode_jpeg(data)
            except ValueError as e:
                await websocket.send_json({"error": str(e)})
                continue
            try:
                result = _run_alpr(frame)
                await websocket.send_json(result)
                _persist_alpr(
                    frame=frame,
                    jpeg_bytes=data,
                    endpoint="/alpr/ws",
                    client_ip=websocket.client.host if websocket.client else None,
                    result=result,
                )
            except Exception as e:  # keep the stream alive on a single bad frame
                logger.exception("ALPR inference failed")
                await websocket.send_json({"error": f"Inference failed: {e}"})
    except WebSocketDisconnect:
        pass
