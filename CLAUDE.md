# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Next.js + TypeScript web app that runs YOLO object detection on a webcam feed. The default/original mode runs entirely client-side in the browser via ONNX Runtime Web (WASM): the feed is captured to a `<canvas>`, preprocessed into a tensor in JS, run through an ONNX model locally, and the results drawn back onto the canvas as bounding boxes. A second, selectable mode instead posts frames to a server-side inference API — see "Server-side inference" below.

## Commands

```bash
npm install       # or yarn install
npm run dev       # start dev server at http://localhost:3000
npm run build     # production build
npm start         # serve production build
npm run lint      # next lint
```

There is no test suite configured in this repo.

## Architecture

**Data flow:** `pages/index.tsx` → `components/models/Yolo.tsx` → `components/ObjectDetectionCamera.tsx` → `utils/runModel.ts` → back up through `postprocess` to draw on canvas.

- **`components/models/Yolo.tsx`** is the core of the app. It owns:
  - `RES_TO_MODEL`: the ordered list of `[resolution, modelFilename]` pairs the "Change Model" button cycles through. Adding a new model means adding an entry here.
  - `preprocess`: resizes the captured canvas frame to the model's input resolution, converts RGBA image data into a normalized `[1,3,W,H]` float32 `Tensor` (NCHW, values /255).
  - `postprocessMap`: dispatches to a per-model postprocess function (`postprocessYolov10`, `postprocessYolov7`) because different YOLO export formats have different output tensor shapes:
    - YOLOv7-tiny: `[det_num, 7]` — `[batch_id, x0, y0, x1, y1, cls_id, score]` per row, already NMS'd by the model.
    - YOLOv10/YOLOv11/YOLOv12: `[1, 300, 6]` — `[x0, y0, x1, y1, score, cls_id]`, zero-padded to 300 rows, pre-sorted descending (loop breaks on first score < 0.25). YOLOv10's ultralytics export bakes NMS into the graph by default; YOLOv11/12's don't unless exported with `nms=True` (see `convert_pt_to_onnx/ultralytics_pt_to_onnx.md`'s "Preferred export" section) — this app's `models/yolo11n.onnx`/`yolo12n.onnx` are exported that way specifically so they share `postprocessYolov10` instead of needing their own NMS/IoU implementation in JS.
  - When adding a new model, you generally need to add both a new `RES_TO_MODEL` entry and, unless its output format matches an existing one (check by exporting with `nms=True` first if it's a raw-output YOLO variant), a new postprocess function plus a `postprocessMap` entry.

- **`components/ObjectDetectionCamera.tsx`** is model-agnostic UI/plumbing: manages the webcam (`react-webcam`), the overlay `<canvas>`, live-detection loop (`requestAnimationFrame`), single-shot capture, camera switching, and timing/FPS stats. It receives `preprocess`/`postprocess` as props from `Yolo.tsx` and has no YOLO-specific knowledge.

- **`utils/runModel.ts`**: thin wrapper around `onnxruntime-web`. `createModelCpu` loads an `InferenceSession` with the `wasm` execution provider; `runModel` feeds the tensor in and times the inference.

- **`data/yolo_classes.ts`**: the 80 COCO class labels, indexed by `cls_id` from model output.

- **Model files live in `/models/*.onnx`** and are copied into `public/runtime/` at build time via a webpack `CopyPlugin` config in `next.config.js` (along with the ONNX Runtime `.wasm` binaries). Models are loaded at runtime via a fetch from `/runtime/<modelName>` — this is why new model files must be dropped in `/models/` (the CopyPlugin picks them up automatically). `public/runtime/` (not `/_next/static/`) is deliberate: these files are unhashed, so their URL doesn't change when content does (e.g. an onnxruntime-web version bump), but Next.js hardcodes an unconditional immutable 1-year `Cache-Control` for the entire `/_next/static/` tree with no override possible — that mismatch is what caused a stale cached wasm binary to link against new JS glue code after a version upgrade. `public/runtime/` gets an explicit `no-cache` header instead (see `headers()` in `next.config.js`), so browsers always revalidate.

- **PWA**: `next-pwa` wraps the Next config (`next.config.js`) to generate a service worker into `public/` for offline installability.

## Server-side inference (alternative to the client-side WASM path)

`components/models/YoloServer.tsx` is a second implementation of the same UI, selectable via a toggle on `pages/index.tsx`, that runs inference on a server instead of in the browser:

- **Why**: the client-side path needs a one-time ~10-25MB download (WASM runtime + model) before it can detect anything. Over a slow/relayed connection (e.g. taco served via Tailscale Funnel, which relays all traffic through Tailscale's infrastructure rather than a direct connection) that download can take minutes or fail outright. Server-side inference replaces that one-time large download with a small recurring per-frame JPEG upload (tens of KB) instead.
- **`inference-server/`**: a standalone FastAPI service (Python, not part of the Next.js app) that loads the same `/models/*.onnx` files via `onnxruntime` (CPU) and exposes `POST /predict?model=<name>` (raw JPEG body in, JSON detections out) and `GET /health`. `inference-server/postprocess.py` is a deliberate line-for-line Python port of `Yolo.tsx`'s per-model postprocess functions (NMS, IoU, decode) so detection behavior matches the client-side version exactly. Detections are returned normalized to `[0,1]` (fraction of the model's input resolution) rather than pixel coordinates, so the client just multiplies by its own canvas size — no need to send resolution back and forth.
- **`YoloServer.tsx`** mirrors `Yolo.tsx`'s loading-state contract (`ready`/`sessionError`/retry-with-backoff via `ObjectDetectionCamera`) but checks server reachability (`GET /health`) instead of downloading anything — much shorter timeouts are appropriate here.
- **`ObjectDetectionCamera.tsx`** is intentionally inference-mode-agnostic: it takes a single `detect: (ctx) => Promise<inferenceTimeMs>` prop that fully owns one detection pass, plus a generic `ready`/loading-state contract. Both `Yolo.tsx` (local) and `YoloServer.tsx` (remote) implement `detect` differently but share all the camera/UI plumbing.
- **Deployment**: `inference-server` runs as its own systemd service on taco (`object-detection-inference.service`, `uvicorn main:app`), independent of the Next.js app's own deployment. Mounted at `taco.tail9f615d.ts.net:8443/infer` (path-based, alongside llama-swap and notify-proxy on the same Funnel port — Tailscale Funnel only exposes 3 public ports total, so new services share existing ports via path routing rather than claiming new ports).
- **Not yet done**: GPU acceleration (taco has ROCm already set up for llama.cpp, but the inference server currently uses `CPUExecutionProvider` — CPU inference is already fast enough for the nano models at ~15-35ms, so this hasn't been a priority). License plate detection/OCR is now done via the pretrained `fast-alpr` package rather than the custom-trained model `docs/PLAN-realtime-license-plate-detection.md` planned — see "License plate detection + OCR" below; that plan doc's own dataset/fine-tuning work (`data/license_plates/`) is still unused/optional.

### License plate detection + OCR

`inference-server/alpr.py` adds plate detection + OCR using the pretrained [`fast-alpr`](https://github.com/ankandrew/fast-alpr) package (YOLOv9 detector + CCT/OCR model, both ONNX, downloaded from `fast-alpr`'s model hub on first use and cached under `~/.cache/{open-image-models,fast-plate-ocr}/`) instead of the custom detector trained from `data/license_plates/`. Endpoints: `GET /alpr/health`, `POST /alpr/predict` (single frame, JPEG in/JSON out, mirrors `/predict`), and `WS /alpr/ws` (persistent stream — see below). `main.py` mounts this router via `app.include_router(alpr_router)`.

- **Why a WebSocket instead of per-frame POST or WebRTC**: this endpoint is meant to be driven at ~1 fps (OCR is more expensive than plain detection, and there's no benefit reading the same plate faster than once a second). At that rate a video codec (WebRTC) buys nothing — every frame is effectively a keyframe, so it only adds SDP/ICE/TURN complexity for zero bandwidth benefit — and a plain per-frame POST re-pays a request/response round trip every second for no reason when a single persistent connection can amortize that once and let the server push results back as they're ready. Full reasoning is in `alpr.py`'s module docstring.
- **`components/models/AlprServer.tsx`** is the client: opens one `WebSocket` to `/alpr/ws` after an `/alpr/health` check (same retry-with-backoff shape as `YoloServer.tsx`), then implements the same `detect: (ctx) => Promise<inferenceTimeMs>` contract `ObjectDetectionCamera.tsx` expects — self-pacing to a 1s minimum interval between sends inside `detect()` regardless of how fast `ObjectDetectionCamera`'s live-detection loop calls it. Selectable as a third mode ("License plates (server, ~1 fps)") on `pages/index.tsx`. Request/response correlation over the WebSocket relies on the server answering in strict send order (a FIFO queue of pending promise resolvers on the client, no request IDs needed) — safe because `alpr.py`'s `receive_bytes`/`send_json` loop is sequential per-connection.
- **Response shape** (both `/alpr/predict` and `/alpr/ws`): `{inferenceTimeMs, detections: [{box: [x0,y0,x1,y1] normalized 0-1, detectionConfidence, plate, ocrConfidence, region, regionConfidence}]}` — `plate`/`region` are `null` if OCR found no text for that box.

## Adding a custom model

Per `convert_pt_to_onnx/ultralytics_pt_to_onnx.md` and the README:

1. Export from ultralytics: `model.export(format="onnx", simplify=True, dynamic=True)`.
2. Convert/optimize for `onnxruntime-web` WASM compatibility: `python -m onnxruntime.tools.convert_onnx_models_to_ort <model>.onnx --save_optimized_onnx_model` (a raw ultralytics ONNX export often throws a `protobuf` error under onnxruntime-web without this step).
3. Place the resulting `.onnx` file in `/models/`.
4. Add an entry to `RES_TO_MODEL` in `components/models/Yolo.tsx`, and add/select a matching entry in `postprocessMap` based on the model's actual output tensor shape (inspect it — don't assume it matches an existing model family).
