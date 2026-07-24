# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Next.js + TypeScript web app that runs YOLO object detection entirely client-side, in the browser, using ONNX Runtime Web (WASM execution provider). No backend inference — the webcam feed is captured to a `<canvas>`, preprocessed into a tensor in JS, run through an ONNX model, and the results are drawn back onto the canvas as bounding boxes.

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
  - `postprocessMap`: dispatches to a per-model postprocess function (`postprocessYolov12`, `postprocessYolov11`, `postprocessYolov10`, `postprocessYolov7`) because different YOLO export formats have different output tensor shapes:
    - YOLOv7-tiny: `[det_num, 7]` — `[batch_id, x0, y0, x1, y1, cls_id, score]` per row, already NMS'd by the model.
    - YOLOv10: `[1, num_boxes, 6]` — `[x0, y0, x1, y1, score, cls_id]`, already NMS'd, scores are pre-sorted descending (loop breaks on first score < 0.25).
    - YOLOv11/YOLOv12: `[1, 84, num_anchors]` — raw output (4 box coords + 80 class scores per anchor), **not** NMS'd by the model, so this app applies its own NMS (`applyNMS`/`calculateIoU`) and confidence filtering (0.25 threshold) in JS.
  - When adding a new model, you generally need to add both a new `RES_TO_MODEL` entry and, unless its output format matches an existing one, a new postprocess function plus a `postprocessMap` entry.

- **`components/ObjectDetectionCamera.tsx`** is model-agnostic UI/plumbing: manages the webcam (`react-webcam`), the overlay `<canvas>`, live-detection loop (`requestAnimationFrame`), single-shot capture, camera switching, and timing/FPS stats. It receives `preprocess`/`postprocess` as props from `Yolo.tsx` and has no YOLO-specific knowledge.

- **`utils/runModel.ts`**: thin wrapper around `onnxruntime-web`. `createModelCpu` loads an `InferenceSession` with the `wasm` execution provider; `runModel` feeds the tensor in and times the inference.

- **`data/yolo_classes.ts`**: the 80 COCO class labels, indexed by `cls_id` from model output.

- **Model files live in `/models/*.onnx`** and are copied into `public/runtime/` at build time via a webpack `CopyPlugin` config in `next.config.js` (along with the ONNX Runtime `.wasm` binaries). Models are loaded at runtime via a fetch from `/runtime/<modelName>` — this is why new model files must be dropped in `/models/` (the CopyPlugin picks them up automatically). `public/runtime/` (not `/_next/static/`) is deliberate: these files are unhashed, so their URL doesn't change when content does (e.g. an onnxruntime-web version bump), but Next.js hardcodes an unconditional immutable 1-year `Cache-Control` for the entire `/_next/static/` tree with no override possible — that mismatch is what caused a stale cached wasm binary to link against new JS glue code after a version upgrade. `public/runtime/` gets an explicit `no-cache` header instead (see `headers()` in `next.config.js`), so browsers always revalidate.

- **PWA**: `next-pwa` wraps the Next config (`next.config.js`) to generate a service worker into `public/` for offline installability.

## Adding a custom model

Per `convert_pt_to_onnx/ultralytics_pt_to_onnx.md` and the README:

1. Export from ultralytics: `model.export(format="onnx", simplify=True, dynamic=True)`.
2. Convert/optimize for `onnxruntime-web` WASM compatibility: `python -m onnxruntime.tools.convert_onnx_models_to_ort <model>.onnx --save_optimized_onnx_model` (a raw ultralytics ONNX export often throws a `protobuf` error under onnxruntime-web without this step).
3. Place the resulting `.onnx` file in `/models/`.
4. Add an entry to `RES_TO_MODEL` in `components/models/Yolo.tsx`, and add/select a matching entry in `postprocessMap` based on the model's actual output tensor shape (inspect it — don't assume it matches an existing model family).
