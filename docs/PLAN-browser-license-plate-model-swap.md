# Plan: swap the browser's ONNX model for a license-plate-aware one

## Goal

Replace (or augment) `Yolo.tsx`'s in-browser detector — currently always
one of the 80-COCO-class models in `RES_TO_MODEL` (`yolo12n.onnx`,
`yolo11n.onnx`, `yolov10n.onnx`, three `yolov7-tiny` sizes) — with one that
also recognizes license plates, so plate *boxes* can be drawn client-side
without a round trip to the server.

**Scope note up front**: this only gets you plate *boxes* in the browser,
not plate *text*. Reading the text (OCR) is a separate model with its own
size/latency cost — see "OCR stays server-side" below. Don't let "replace
the object detector" quietly turn into "replace the ALPR pipeline"; those
are different features.

## Current state (verified, not assumed)

- **No existing artifact is a drop-in replacement.** The one candidate
  already trained — `runs/detect/runs/license_plate/
  license_plate_ft_all_categories/weights/best.onnx` (from
  `training/finetune_license_plate_interleaved.py`) — was inspected
  directly with `onnxruntime` for this plan:
  - Output shape: `[1, 85, 1344]` = 4 box coords + 81 class scores ×
    1344 anchors. This is the **raw ultralytics detection head**, not
    NMS'd or decoded.
  - `names` metadata confirms 81 classes: the original 80 COCO classes
    (indices 0-79, same order as `data/yolo_classes.ts`) plus
    `license_plate` at index 80. So this is an **augmented** model
    (COCO + plates), not a plate-only one — training added the class
    without removing the others.
  - This raw shape is **not compatible with either existing client
    postprocess function**: `postprocessYolov10` expects pre-sorted,
    already-NMS'd `[1,300,6]` (score/cls_id already resolved per row);
    `postprocessYolov7` expects already-NMS'd `[det_num,7]`. Neither
    does the anchor-decode + IoU-suppression this raw output still needs.
    Per CLAUDE.md, the existing models sidestep this entirely by
    exporting with `nms=True` baked into the ONNX graph — this
    checkpoint wasn't exported that way.
  - Per `docs/REPORT-license-plate-finetuning.md`, training on this
    checkpoint was paused mid-run to land bugfixes; treat its accuracy
    as unvalidated, not production-ready, regardless of the export issue.
- **The server-side alternative (`fast-alpr`'s detector,
  `yolo-v9-t-384-license-plate-end2end`) is plate-only**, pretrained,
  downloaded from `open-image-models`'s hub, and already proven in
  production via `inference-server/alpr.py`. It has never been exported
  for `onnxruntime-web`/WASM — only ever run server-side via `onnxruntime`
  (CPU). Its class list is just `{0: "license_plate"}`, so it can't
  replace a COCO detector without losing the other 79 classes entirely.
- **`data/yolo_classes.ts`** is a flat 80-entry array indexed by `cls_id`,
  shared by every current model. Every postprocess function hardcodes
  `yoloClasses[cls_id]` — there's no per-model class list parameter today.

## Two options

### Option A (recommended): augment, don't replace

Use (a re-exported version of) the 81-class `all_categories` checkpoint,
or re-run/re-validate that fine-tune to completion first. Ship it as a
**new `RES_TO_MODEL` entry** alongside the existing six, not as the
default — exactly the existing "Change Model" cycle-through UX, so users
who want plate detection can select it and everyone else is unaffected.

Pros: zero regression risk to existing COCO detection; matches how every
other model was added (per CLAUDE.md's "Adding a custom model"); can be
promoted to default later once field-tested.

Cons: 81-class model may be a little larger/slower than the 80-class
nanos (rough parity expected, not verified); accuracy for `license_plate`
specifically is unvalidated (see "Current state" above) and likely worse
than the dedicated `fast-alpr` detector, which was trained specifically
and only for plates.

### Option B: true replacement (plate-only)

Export `fast-alpr`'s `yolo-v9-t-384-license-plate-end2end` (or train a new
plate-only nano) for browser use, and swap it in for one of the existing
`RES_TO_MODEL` slots. The app would detect *only* plates in that mode,
losing car/person/etc. detection while it's selected.

Pros: reuses the detector already proven accurate in production (the same
one `alpr.py` runs), no need to re-validate a paused/unfinished fine-tune.

Cons: a genuine capability regression for that mode (an app called
"Real-Time Object Detection" that only detects one class is a strange
default); `data/yolo_classes.ts`'s shared 80-entry array assumption breaks
(a plate-only model needs its own 1-entry class list, requiring the small
parameterization change noted below regardless of which option is chosen).

**Recommendation: do Option A.** It's strictly additive, reuses the
existing per-model-family postprocess dispatch, and doesn't ask users to
trade away existing functionality for a still-unvalidated plate detector.
Revisit Option B only if Option A's accuracy turns out to be unacceptably
worse than server-side `fast-alpr` and a plate-focused mode is wanted
regardless of the COCO-class tradeoff.

## Required steps (Option A)

1. **Finish/re-validate the fine-tune.** Per
   `docs/REPORT-license-plate-finetuning.md`, resume the paused training
   run (or start fresh from `license_plate_ft_interleaved`) through to a
   checkpoint whose `license_plate` mAP is actually acceptable — don't
   ship the current unfinished checkpoint as-is.
2. **Re-export with NMS baked in**: `model.export(format="onnx",
   simplify=True, dynamic=True, nms=True)` from the resulting `.pt` (see
   CLAUDE.md's "Adding a custom model" step 1) — this changes the output
   to the same `[1,300,6]` shape `postprocessYolov10` already handles, so
   no new postprocess function should be needed as long as `cls_id` still
   indexes correctly into an 81-entry class list.
3. **ORT-convert for WASM compatibility**: `python -m onnxruntime.tools.
   convert_onnx_models_to_ort <model>.onnx --save_optimized_onnx_model`
   (step 2 of the same CLAUDE.md workflow) — skipping this is what causes
   the documented `protobuf` error under `onnxruntime-web`.
4. **Extend `data/yolo_classes.ts`** with `"license_plate"` as index 80,
   *only* for this model — since every other model's `cls_id` still maps
   into the original 80-entry array, either:
   - confirm training preserved COCO's exact class order for indices
     0-79 (it does, per the inspected metadata above) and simply append
     `"license_plate"` to the existing shared array (simplest — safe
     because indices 0-79 are unchanged for the other models too), or
   - if that ever stops being true for a future export, parameterize
     `postprocessYolov10`/`postprocessYolov7`'s hardcoded
     `yoloClasses[cls_id]` lookups to take a class-list argument instead.
5. **Add the `RES_TO_MODEL` entry** in `Yolo.tsx` (resolution + filename)
   and drop the `.onnx` file in `/models/` — CopyPlugin picks it up
   automatically for both the browser build and the server's `MODELS_DIR`
   (see `docs/system-architecture.md`) if you also want it selectable
   server-side via `/predict?model=...`.
6. **Update `utils/notify.ts`**: `CLASS_TO_TOPIC`'s comment currently
   says plates have "no model that currently detects them" — once this
   ships, either map `license_plate` to a topic or explicitly note why
   it's still excluded (e.g. plate detections without OCR aren't
   actionable enough to page someone about).
7. **Reconsider `utils/autoAlprSubmit.ts`**: its whole premise is "the
   browser can't see plates, so forward car/person frames to the server's
   plate detector opportunistically." Once the browser detects plates
   directly, that heuristic is redundant for *detection* — but a server
   round trip is still the only way to get OCR text (see below), so the
   trigger classes may want to change from `{car, person}` to
   `{license_plate}` (only forward frames where a plate box was actually
   found) rather than removing the auto-submit path entirely.

## OCR stays server-side (out of scope for this swap)

This plan only covers the *detector*. Reading plate text requires a
second model (`fast-plate-ocr`'s CRNN, what `alpr.py` already runs) —
`docs/PLAN-realtime-license-plate-detection.md` §4 already looked at
bringing that in-browser and the size/latency cost was the reason
`fast-alpr` ended up server-side-only in the first place. Nothing here
changes that conclusion: a browser plate detector gives you boxes to draw
locally and, combined with the `autoAlprSubmit` update above, a sharper
trigger for *when* to forward a frame for OCR — not in-browser OCR itself.

## Testing/rollout

1. Ship as a new, non-default `RES_TO_MODEL` entry (Option A, step 5).
2. Manually verify on a handful of real phone captures (varied angle/
   distance/lighting) before considering it more than experimental —
   the underlying fine-tune has no reported validation metrics yet.
3. Only after field use looks solid, consider promoting it to
   `RES_TO_MODEL[0]` (the default loaded on first visit).
