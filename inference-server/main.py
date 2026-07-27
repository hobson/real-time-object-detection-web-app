"""FastAPI server that runs YOLO ONNX inference server-side.

Exists as an alternative to the client-side onnxruntime-web path: the
browser just captures a JPEG frame and posts it here instead of downloading
a ~20MB wasm runtime + model to run inference locally. See CLAUDE.md
("Server-side inference") for the tradeoffs.
"""
import io
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from alpr import router as alpr_router
from limits import validate_body_size
from persist import persist_submission
from postprocess import POSTPROCESS_MAP
from request_parsing import parse_image_and_metadata

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent.parent / "models"))

RES_TO_MODEL = {
    "yolo12n.onnx": (256, 256),
    "yolo11n.onnx": (256, 256),
    "yolov10n.onnx": (256, 256),
    "yolov7-tiny_256x256.onnx": (256, 256),
    "yolov7-tiny_320x320.onnx": (320, 320),
    "yolov7-tiny_640x640.onnx": (640, 640),
}
DEFAULT_MODEL = "yolo12n.onnx"

_sessions: dict[str, ort.InferenceSession] = {}


def get_session(model_name: str) -> ort.InferenceSession:
    # No lock: this server runs as a single uvicorn worker, and this function
    # has no `await`, so a coroutine calling it always runs to completion
    # before another gets the event loop - no concurrent-init race is possible.
    if model_name not in _sessions:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Unknown model {model_name}")
        _sessions[model_name] = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
    return _sessions[model_name]


app = FastAPI(title="YOLO inference server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(alpr_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {"models": list(RES_TO_MODEL.keys()), "default": DEFAULT_MODEL}


@app.post("/predict")
async def predict(request: Request, model: str = Query(DEFAULT_MODEL)):
    """Run general object detection on one image.

    **Request body** - either:
    - raw JPEG/PNG bytes (`Content-Type: image/jpeg`), the original
      contract, e.g. `curl --data-binary @photo.jpg -H "Content-Type:
      image/jpeg" ".../predict"`; or
    - `multipart/form-data` with an `image` file part and an optional
      `metadata` JSON string part carrying GPS/device-orientation/
      acceleration/camera-facing data and/or the client's own YOLO
      detections (key `client_detections`, same shape as this endpoint's
      own `detections` response field) - see docs/user-manual.md.

    Both shapes return the same JSON response.
    """
    if model not in RES_TO_MODEL:
        raise HTTPException(status_code=400, detail=f"Unknown model {model}")

    body, metadata, image_content_type = await parse_image_and_metadata(request)
    validate_body_size(body)
    client_detections = metadata.pop("client_detections", None) if metadata else None

    try:
        image = Image.open(io.BytesIO(body)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    resolution = RES_TO_MODEL[model]
    resized = image.resize(resolution, Image.Resampling.BILINEAR)
    data = np.asarray(resized, dtype=np.float32) / 255.0  # HWC
    tensor = np.transpose(data, (2, 0, 1))[np.newaxis, ...].astype(np.float32)  # NCHW

    session = get_session(model)
    input_name = session.get_inputs()[0].name

    start = time.time()
    output = session.run(None, {input_name: tensor})[0]
    inference_ms = (time.time() - start) * 1000

    detections = POSTPROCESS_MAP[model](output, resolution)

    persist_submission(
        image_bytes=body,
        endpoint="/predict",
        width=image.width,
        height=image.height,
        content_type=image_content_type,
        model_name=model,
        client_ip=request.client.host if request.client else None,
        inference_time_ms=inference_ms,
        status_code=200,
        detections=[
            {"class_name": d["class"], "confidence": d["confidence"], "box": d["box"]}
            for d in detections
        ],
        capture_metadata=metadata,
        client_detections=client_detections,
    )

    return JSONResponse(
        {
            "model": model,
            "inferenceTimeMs": round(inference_ms, 1),
            "detections": detections,
        }
    )
