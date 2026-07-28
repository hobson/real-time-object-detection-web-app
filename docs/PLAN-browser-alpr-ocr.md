# Plan: in-browser license-plate OCR (text reading)

## Goal

Read the actual plate *text* client-side, building on the in-browser plate
*detector* already shipped (`yolo-v9-t-384-license-plate-end2end.onnx` in
`Yolo.tsx`, see the commit that added it). This plan covers porting
`fast-plate-ocr`'s preprocessing, ONNX inference, and decode logic to
JS/WASM so a detected plate box can be read without a round trip to
`/alpr/predict`.

Everything below is grounded in reading `fast-plate-ocr`'s and
`fast-alpr`'s actual source (`inference-server/.venv/lib/*/site-packages/
fast_plate_ocr/core/process.py` and `fast_alpr/alpr.py`) and inspecting
the real cached ONNX model with `onnxruntime`, not assumed from docs.

## What the server-side pipeline actually does (verified)

1. **Crop**: `fast_alpr.alpr.ALPR.predict()` crops the detected plate with
   a plain axis-aligned rectangle - `img[y1:y2, x1:x2]`, no perspective
   warp, no rotation correction. This is the easy case: a browser
   equivalent is one `ctx.drawImage(source, x0, y0, w, h, 0, 0, w, h)`
   onto a fresh canvas.
2. **Preprocess** (`fast_plate_ocr/core/process.py:resize_image` +
   `preprocess_image`): resize the crop to a fixed size from the model's
   config (see below), **no aspect-ratio preservation by default**
   (`keep_aspect_ratio: false` in every shipped config), then... nothing
   else. `preprocess_image`'s own docstring: "the model itself handles
   pixel-value normalisation" - confirmed by inspecting the ONNX model's
   input directly:
   ```
   input: input ['unk__801', 64, 128, 3] tensor(uint8)
   ```
   NHWC, **uint8**, no `/255` or mean/std normalization needed client-side
   - normalization is baked into the ONNX graph itself. This is
   meaningfully simpler than `Yolo.tsx`'s existing detector preprocessing
   (which does need the NCHW transpose + `/255` + float32 cast).
3. **Inference**: one `onnxruntime` call.
4. **Decode** (`fast_plate_ocr/core/process.py:postprocess_output`) -
   confirmed **not CTC**, much simpler than initially assumed:
   ```
   output "plate":  [N, 10, 37]   # 10 character slots x 37-char alphabet
   ```
   Reshape to `(N, max_plate_slots, vocab_size)`, `argmax` over the last
   axis per slot, map each index through the alphabet string, join into a
   string, strip trailing `pad_char`. No blank-collapsing, no beam search
   - a per-slot independent classification, decodable in ~10 lines of JS.
5. **Region classification** (optional, same model): a second output head,
   `output "region": [N, 66]` - another plain `argmax`, mapped through a
   `region_labels` list (66 country/region names from the model's config
   YAML, `"Unknown"` included).

## Concrete model facts (default config, `cct-xs-v2-global-model`)

From `~/.cache/fast-plate-ocr/cct-xs-v2-global-model/*.yaml` and the ONNX
file itself:

| Field | Value |
|---|---|
| ONNX file size | 3.3MB |
| Input | `[N, 64, 128, 3]`, uint8, RGB |
| `max_plate_slots` | 10 |
| `alphabet` | `"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"` (37 chars, `_` = pad) |
| `img_height` / `img_width` | 64 / 128 |
| `keep_aspect_ratio` | false |
| `image_color_mode` | rgb |
| Output `plate` | `[N, 10, 37]` |
| Output `region` | `[N, 66]` |

3.3MB is small - comparable to or smaller than the WASM runtime files
already downloaded for the COCO detectors, so download size is not the
blocker this plan needs to worry about the way it was for a full OCR
pipeline evaluated in `docs/PLAN-realtime-license-plate-detection.md` §4.
That earlier rejection was made without this level of detail on the
model's actual size/preprocessing simplicity - worth the fresh look this
plan represents.

## Implementation outline

1. **Export/convert**: same treatment as the detector -
   `onnxruntime.tools.convert_onnx_models_to_ort` on the cached
   `cct_xs_v2_global.onnx` for WASM compatibility, drop into `/models/`
   (e.g. `cct-xs-v2-global-ocr.onnx`).
2. **Crop step** in `Yolo.tsx` (or a new small helper): once
   `postprocessYolov7` (the plate detector's postprocess function) has a
   box, draw that sub-rectangle of the *original* captured frame - not
   the boxed/annotated canvas - onto an offscreen canvas.
3. **OCR preprocess**: resize the crop to 128x64 (no aspect preservation,
   matches server behavior), read pixels via `getImageData`, drop the
   alpha channel to get `[64,128,3]` `Uint8Array` - no normalization, no
   NCHW transpose (this model wants NHWC uint8 as-is). Meaningfully less
   code than `Yolo.tsx`'s existing `preprocess()`.
4. **Inference**: a second `onnxruntime-web` `InferenceSession`, run
   after the detector's session, only on frames where a plate box exists
   above the confidence threshold (no benefit running OCR on frames with
   no detected plate).
5. **Decode** (new, small, pure-JS function - no existing analog in this
   codebase to reuse, unlike detection postprocessing):
   ```ts
   function decodePlate(output: Float32Array, slots: number, alphabet: string, padChar: string): string {
     let text = '';
     for (let s = 0; s < slots; s++) {
       const row = output.slice(s * alphabet.length, (s + 1) * alphabet.length);
       const idx = row.indexOf(Math.max(...row)); // argmax
       text += alphabet[idx];
     }
     return text.replace(new RegExp(`${padChar}+$`), '');
   }
   ```
   Region decode is the same pattern against the second output, one more
   `argmax` over 66 values.
6. **Display**: draw the decoded text near the plate box, same visual
   pattern `AlprServer.tsx` already uses for the server-side mode.
7. **Wire into existing infra**: this is a natural point to revisit
   `utils/autoAlprSubmit.ts`'s trigger classes (see
   `docs/PLAN-browser-license-plate-model-swap.md`'s open item) - once
   both detection *and* OCR run fully client-side, the whole rationale
   for forwarding frames to `/alpr/predict` changes from "get plate
   detection the browser can't do" to "get a second opinion / log to
   `/admin`," which is a different feature than what auto-submit
   currently does.

## Testing plan

1. **Unit-style JS test of the decode function alone**, before any ONNX
   involvement: feed it a known `[10,37]` array (e.g. one-hot argmax
   positions spelling "ABC1234_​__"), assert it returns `"ABC1234"`.
   Cheap, catches indexing/transposition bugs before they're tangled up
   with canvas/WASM debugging.
2. **Python-side ground truth for comparison**: run the same crop through
   `fast_plate_ocr`'s actual Python `postprocess_output()` (already
   installed in `inference-server/.venv`) on a handful of
   `data/license_plates/images/*.jpg` crops, record the expected decoded
   strings. This is the existing, trusted implementation - use it as the
   oracle the JS port must match, not just "does it look plausible."
3. **Cross-check ONNX output numerically**: run the *same* cropped+resized
   image through both `onnxruntime` (Python, via the oracle script above)
   and `onnxruntime-web` (a small standalone browser test page, not
   wired into `Yolo.tsx` yet) and diff the raw `[10,37]`/`[66]` float
   arrays before trusting the decode step - isolates "is my preprocessing
   right" from "is my decode right" as two independently-checkable
   claims, rather than only judging by whether the final string looks
   like a plate.
4. **End-to-end on real captures**: once wired into `Yolo.tsx`, test
   against `data/license_plates/images/*.jpg` (loaded into a `<canvas>`
   via a file input, bypassing the webcam for repeatable testing) and
   against a live phone camera at varying distance/angle/lighting -
   compare decoded text to `/alpr/predict`'s server-side OCR on the exact
   same frame (send the same captured JPEG to both paths) as a running
   accuracy check, not just a one-time validation.
5. **Confidence-threshold calibration**: `postprocess_output`'s
   `return_confidence` path gives a per-character `argmax` probability -
   surface this in the JS port too and decide a display threshold (e.g.
   don't show OCR text if the mean per-character confidence is low),
   mirroring how `MIN_CONFIDENCE` already gates detection boxes.

## Risks / open questions

- **Region classification's 66-entry label list** must be extracted from
  the model's config YAML (`plate_regions` field) and shipped alongside
  the model as a small JSON/TS array - easy to get wrong if hand-copied;
  generate it programmatically from the YAML once, don't retype it.
- **Multiple OCR model variants exist** (`cct-xs-v2`, `cct-s-v2`,
  `argentinian-plates-*`, `european-plates-*`, ...) with different
  `img_height`/`img_width`/`alphabet`/`max_plate_slots` per model - the
  decode function must take these as parameters (like `Yolo.tsx`'s
  `MODEL_CLASSES` pattern for the detector), not hardcode the global
  model's config, if more than one OCR model is ever offered.
- **WASM protobuf-compatibility check is still needed** for the OCR
  model specifically (step 1) - the detector model needed the ORT
  conversion step to avoid the documented protobuf error; assume the OCR
  model does too until verified, don't skip that step.
- **This is strictly additive to, not a replacement for, server-side
  OCR** - `/alpr/predict` remains the source of truth for anything
  persisted to `/admin` (curation, training-data review), since only the
  server path writes to Postgres. In-browser OCR is for instant/offline
  display only.
