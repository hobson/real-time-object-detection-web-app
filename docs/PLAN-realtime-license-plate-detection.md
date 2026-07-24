# License plate detection + reading (OCR): dataset + fine-tuning plan

This plan targets **in-browser inference first** (the app's default,
original mode — see [`realtime-object-detection.md`](./realtime-object-detection.md))
so plate detection/reading works the same way the existing 80 COCO
classes do: entirely client-side, no data leaving the phone unless
notifications are enabled. Where the browser path has real added cost or
complexity (mainly: OCR model size, see §4), an **optional** taco
server-side alternative is noted — see
[`realtime-object-detection.md` §4](./realtime-object-detection.md#4-pick-a-mode-in-browser-vs-server-side)
for the existing In-browser/Server-side toggle this would plug into.

The dataset itself lives in `data/license_plates/` (images gitignored,
~221MB, re-downloadable — see §1); only this plan document moved to
`docs/`.

## 1. Dataset

**Source:** Google [Open Images V7](https://storage.googleapis.com/openimages/web/index.html), class `/m/01jfm_` ("Vehicle registration plate"), validation split.

Downloaded 724 images (all validation-split images containing at least one
plate annotation; `max_samples=1000` was requested but only 724 exist in
that split). Images + Open-Images-format CSV annotations were fetched
directly from Google's public GCS bucket (no account/API key needed) via
`fiftyone.zoo.load_zoo_dataset(...)`. FiftyOne's own MongoDB-backed
indexing isn't available in this environment (no `mongod`), so the raw
downloaded files were converted to YOLO format directly instead — see
`/tmp/.../convert_oiv7_plates.py` in this session's scratch dir for the
conversion script (not committed; trivial to regenerate, ~40 lines).

**Layout** (`data/license_plates/`):
```
images/       724 .jpg files
labels/       724 .txt files, YOLO format (class_id cx cy w h, normalized 0-1)
classes.txt   single line: "license_plate"
dataset.yaml  ultralytics-compatible dataset config
train.txt     651 image paths (90%)
val.txt       73 image paths (10%), seed=42
```

Single class (`0 = license_plate`), boxes normalized directly from Open
Images' `XMin/XMax/YMin/YMax` (already 0-1) to YOLO's center-x/center-y/w/h
convention. Spot-checked visually — box placement is correct.

**Licensing — read before using beyond local experimentation:**
- Annotations: CC BY 4.0 (Google).
- Images: CC BY 2.0, sourced from Flickr.
- Both permit commercial use *with attribution*, but Google's own docs
  explicitly disclaim warranty on a per-image basis: "we make no
  representations or warranties regarding the license status of each
  image and you should verify the license for each image yourself."
- **Practical implication:** fine-tuning a model on this data for local/
  personal use (this project's actual use case) is low-risk. Redistributing
  the *dataset itself*, or shipping a *commercial* product trained on it,
  would technically require per-image attribution back to individual Flickr
  photographers — not practical at 724-image scale. If this ever goes
  beyond personal use, either build an attribution manifest (Open Images
  publishes photographer/source metadata alongside the CSVs) or switch to
  a dataset with a cleaner blanket license (e.g. a purpose-built ALPR
  dataset with its own explicit license, several of which exist on
  Roboflow Universe — those need a free Roboflow account/API key to
  download programmatically, which is why Open Images was used instead
  for this first pass with no external account requirement).

**724 images is a starting point, not a finished dataset.** For a
production-quality detector you'd want low thousands of images minimum,
more geographic/lighting/angle diversity than one Open Images split
provides, and ideally some images from the actual camera/lens this app
runs on. Treat this as enough to validate the fine-tuning *process*
end-to-end, then decide whether to invest in a bigger dataset once the
approach is proven.

## 2. Adding `license_plate` as a model class

Two approaches, in order of recommendation:

### Option A (recommended): add as a new 81st class

Fine-tune with `nc=81` (COCO's 80 + `license_plate`), rather than
overwriting an existing class slot. Ultralytics' training pipeline
supports this directly by editing the model's class head to add one
output, initialized fresh (or warm-started — see §3) for the new class
while the other 80 keep their pretrained weights.

**Why this over swapping:** every piece of code in this repo currently
assumes a fixed 80-class COCO output (`data/yolo_classes.ts`,
`inference-server/yolo_classes.py`, `NUM_CLASSES = 80` in both
`postprocess.py` and the client-side `postprocessYolov11`/`applyNMS`
logic in `Yolo.tsx`). Adding a class means touching those in one place
each (append to the class list, bump `NUM_CLASSES`/`4 + NUM_CLASSES`
slicing) — mechanical, low-risk, and nothing else silently breaks. It also
keeps all 80 original COCO classes intact, so this app doesn't regress on
person/car/etc. detection to gain plate detection.

### Option B: repurpose an underused existing class slot

Keep `nc=80` fixed and retrain one existing class index (e.g. `77 hair
drier` — a class this app will realistically never need) to mean
`license_plate` instead. Zero code changes to tensor shapes/slicing
anywhere.

**Why not this by default:** it permanently loses whatever class you
repurpose (fine, for `hair drier`; less fine if you later want it back),
and it's a one-way door baked into the exported `.onnx` file — the class
index `77` means something different in this model than in every other
`yolo*.onnx` file in `/models/`, which is an easy source of confusing bugs
later (e.g. if someone runs the old and new models side by side, or
copy-pastes a class-index constant between them). Reasonable choice if you
specifically want to avoid touching `NUM_CLASSES` everywhere, but Option A
is cleaner given how few places actually reference it in this codebase.

## 3. Fine-tuning process, warm-started from `car` (or `stop sign`)

The core idea: a randomly-initialized new class head has to learn "what a
license plate looks like" from nothing, which needs more data/epochs than
this 724-image dataset comfortably provides. Warm-starting the new class's
final-layer weights from a *semantically related* existing class gives the
optimizer a much better starting point — the "car" class's detection head
already encodes useful low/mid-level features (edges, rectangular shapes,
metallic/reflective surfaces, road-scene context) that transfer well to
plates, which are small rigid rectangles that always appear attached to a
vehicle. `stop sign` is the second-best candidate (also a small rigid
rectangle-ish/octagon detected against varied backgrounds), worth trying
as an alternative if `car`-init underperforms.

Using `ultralytics` (the same library referenced in
`convert_pt_to_onnx/ultralytics_pt_to_onnx.md` for this repo's existing
models) rather than raw `onnxruntime`, since class-head surgery on a
`.pt` checkpoint is straightforward with its `nn.Module` structure and
painful directly on an exported `.onnx` graph.

### Step-by-step

1. **Get a `.pt` checkpoint**, not an `.onnx` export — need the trainable
   PyTorch model. Pull the matching pretrained weights for whichever
   architecture this fine-tune targets, e.g. `yolo12n.pt` (matches
   `models/yolo12n.onnx`, this app's current default model).

2. **Expand the class head from 80 to 81 outputs**, copying the `car`
   class's weight row into the new `license_plate` slot as its
   initialization (rather than the framework's default random init):

   ```python
   from ultralytics import YOLO
   import torch

   model = YOLO("yolo12n.pt")
   detect_head = model.model.model[-1]  # ultralytics Detect head
   car_idx = list(COCO_CLASSES).index("car")  # = 2

   for cv3_branch in detect_head.cv3:  # one per detection scale
       final_conv = cv3_branch[-1]  # nc-wide 1x1 conv, one output channel per class
       old_weight, old_bias = final_conv.weight, final_conv.bias
       new_conv = torch.nn.Conv2d(
           old_weight.shape[1], old_weight.shape[0] + 1, kernel_size=1
       )
       with torch.no_grad():
           new_conv.weight[: old_weight.shape[0]] = old_weight
           new_conv.bias[: old_bias.shape[0]] = old_bias
           # Warm-start license_plate (new last index) from car's weights,
           # scaled down so it starts as "car-ish but uncertain" rather
           # than "as confident as car" - avoids the new class dominating
           # early training purely from init, not learned signal.
           new_conv.weight[-1] = old_weight[car_idx] * 0.5
           new_conv.bias[-1] = old_bias[car_idx] - 2.0  # lower initial confidence
       cv3_branch[-1] = new_conv

   detect_head.nc = 81
   ```

   (Exact module path (`model.model.model[-1]`, `cv3` naming) needs
   verifying against the installed `ultralytics` version's `Detect` head
   implementation before running — this sketches the *shape* of the
   surgery, not a copy-paste-ready script. Check
   `model.model.model[-1].__dict__` interactively first.)

3. **Freeze the backbone, fine-tune the head first.** With only ~650
   training images, full fine-tuning risks catastrophic forgetting of the
   other 80 classes. Freeze everything except the detection head
   (`model.train(..., freeze=[0,1,2,...])` up through the backbone/neck
   layers) for the first pass, confirm `license_plate` starts detecting
   *something* without other classes regressing, then optionally unfreeze
   for a low-LR full fine-tune pass if plate accuracy needs more capacity
   than the head alone can provide.

4. **Train:**
   ```bash
   yolo detect train \
     model=<the-surgically-modified-checkpoint>.pt \
     data=data/license_plates/dataset.yaml \
     epochs=50 imgsz=256 batch=16 \
     freeze=10 \
     project=runs/license_plate lr0=0.001
   ```
   `imgsz=256` matches this app's default model resolution
   (`RES_TO_MODEL[0]` in `Yolo.tsx`) - the model doesn't need to be trained
   at a resolution higher than what it'll actually run at.

5. **Validate per-class**, not just overall mAP — specifically check that
   `car`/`truck`/`bus`/etc. haven't regressed from the class-head surgery
   or fine-tuning, alongside `license_plate`'s own precision/recall.
   `yolo detect val` reports per-class metrics by default.

6. **Export to ONNX** the same way the existing models were produced (per
   `convert_pt_to_onnx/ultralytics_pt_to_onnx.md`):
   ```bash
   python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx', simplify=True, dynamic=True)"
   python -m onnxruntime.tools.convert_onnx_models_to_ort best.onnx --save_optimized_onnx_model
   ```

7. **Wire it into the app — browser path (primary):**
   - Drop the new `.onnx` into `/models/`.
   - Add a `RES_TO_MODEL` entry in `components/models/Yolo.tsx`.
   - Append `"license_plate"` to `data/yolo_classes.ts`, and bump the `80`
     in `NUM_CLASSES`/`4 + NUM_CLASSES` slicing in
     `postprocessYolov11`/`postprocessYolov12` (`Yolo.tsx`) to `81`.
   - Add `license_plate: 'license_plate'` to `CLASS_TO_TOPIC` in
     `utils/notify.ts` if plate detections should also trigger an ntfy
     notification (see
     [`realtime-object-notification.md`](./realtime-object-notification.md)
     — the topic name `object-detection-license_plate` is already reserved
     there, waiting on this model).

   **Optional — also wire into the taco server-side path**, if you want
   plate detection available there too (e.g. for lower-end phones using
   Server-side mode, or once OCR pushes the model past what's comfortable
   to ship to a browser — see §4): the equivalent three edits in
   `inference-server/main.py` (`RES_TO_MODEL`/model list),
   `inference-server/yolo_classes.py` (append the class, kept in sync with
   `data/yolo_classes.ts` manually), and `postprocess_yolov11_12` in
   `inference-server/postprocess.py` (bump `NUM_CLASSES` to 81). Not
   required for the browser path to work — the two inference paths are
   independent and don't have to ship the same model set simultaneously.

### Open question worth deciding before starting

Whether to fine-tune `yolo12n` (this app's current default, opset 19,
needs `onnxruntime-web >= 1.18` per the earlier debugging this session) or
`yolov7-tiny` (older opset, most battle-tested in this app's actual
deployment history). No strong reason to prefer one over the other for
this specific task — pick whichever this app ends up standardizing on
long-term.

## 4. Reading the plate (OCR)

§§1-3 only get a *bounding box* around a plate — "there's a plate here,"
not "the plate says ABC-1234." Actually reading the characters is a
second, independent model/pipeline stage that runs *after* detection,
cropping the detected box and feeding just that crop to an OCR step.
Nothing here has been built yet — this section documents the approach,
same status as the rest of this plan.

### Why this is a separate model from detection

YOLO (detection) answers "where," and is deliberately cheap/fast because
it runs on every frame. Reading characters is a fundamentally different
task — small, often low-contrast/angled text — and needs either a
dedicated OCR architecture or a general-purpose OCR engine. Bolting text
recognition onto the same YOLO output (e.g. as 36 more "classes" for
A-Z/0-9) doesn't work well in practice: character segmentation within a
tiny, perspective-distorted crop is a different problem than "is there a
car-shaped rectangle in this general area," and mixing the two tasks in
one model tends to hurt both.

### Option A (primary target — runs in-browser): a small CRNN plate-OCR model

A **CRNN** (Convolutional Recurrent Neural Network — convolutional
feature extraction feeding an RNN/CTC decoder for the character
sequence) is the standard lightweight architecture for plate OCR
specifically (used across most open ALPR pipelines, e.g. the
`openalpr`/`easyocr`-adjacent projects surfaced in the dataset research
for §1). Small ones (a few MB, similar order of magnitude to the YOLO
nano models already in `/models/`) export to ONNX and run through
**the same `onnxruntime-web` pipeline already built** — no new browser
runtime/architecture needed, just a second model loaded alongside the
detector.

**Pipeline (client-side, `Yolo.tsx`):**
1. Run plate detection as normal (§§1-3) — get a bounding box.
2. Crop that region out of the source canvas (`ctx.getImageData` on the
   box's pixel coordinates, same pattern `preprocess()` already uses for
   the full frame).
3. Resize/normalize the crop to the OCR model's expected input (typically
   a fixed height, variable width for CRNNs — check the specific
   checkpoint's requirements).
4. Run the OCR model's `InferenceSession` on the crop (second
   `runModelUtils.createModelCpu()` call, second small download).
5. CTC-decode the output sequence to a string (greedy decode is fine to
   start — collapse repeated characters, drop the blank token).

**Where to get a starting checkpoint:** pretrained plate-OCR ONNX models
exist publicly (e.g. searches during §1's dataset research surfaced
`morsetechlab/yolov11-license-plate-detection` on Hugging Face and
several ALPR repos bundling a CRNN OCR stage) — worth evaluating an
existing pretrained one before training from scratch, since character
recognition (unlike "what does a license plate look like," which
benefits from this project's own fine-tuning in §§1-3) is a fairly
universal task where public checkpoints often transfer reasonably well
without local fine-tuning. If accuracy on this app's actual camera/plates
isn't good enough, fine-tuning would need its own labeled dataset
(cropped plate images + ground-truth text) — a different, smaller
dataset than §1's, not yet sourced.

**Cost:** a second ~few-MB-to-low-tens-of-MB download, on top of the
detector's. Given this session's whole debugging arc was about a single
~10-25MB download being painful over taco's Tailscale Funnel, a second
model download is a real cost to weigh — pick the smallest CRNN
checkpoint that hits acceptable accuracy, and reuse the existing
retry-with-backoff loading UI (`SESSION_LOAD_TIMEOUT_MS`/`MAX_LOAD_ATTEMPTS`
in `Yolo.tsx`) rather than building new loading UX for it.

### Option B (suggestion — taco server-side): PaddleOCR or EasyOCR

If the in-browser CRNN's accuracy or download-size cost isn't acceptable,
run OCR server-side instead, as an additional stage in
`inference-server/main.py`'s `/predict` handler: detect plates as usual,
crop server-side, run the crop through **PaddleOCR** or **EasyOCR**
(both are full-featured, much more accurate general OCR engines than a
minimal CRNN, at the cost of being much heavier — hundreds of MB of
model weights, meaningfully slower per-inference than the nano YOLO
detectors currently running there). Since taco already runs
`inference-server` as its own systemd service with room to add Python
dependencies, this is a straightforward *addition*, not a new service —
add `easyocr`/`paddleocr` to `inference-server/requirements.txt`, load it
once at startup alongside the existing `onnxruntime.InferenceSession`
objects, and add the crop+OCR step to the response before returning
JSON. The response schema would need a new field per detection (e.g.
`"text": "ABC1234"`) alongside the existing `class`/`confidence`/`box`.

This only benefits **Server-side mode** (§4 of
[`realtime-object-detection.md`](./realtime-object-detection.md)) — it
doesn't help the in-browser path at all, since the OCR would run on
taco, not the phone. Worth doing if server-side becomes the primary mode
for this app, or as a higher-accuracy fallback/comparison against
Option A's lighter in-browser CRNN.

### Recommendation

Start with Option A (in-browser CRNN) to keep this feature working the
same way as the rest of the app (client-side, no data leaving the
phone) — it's the better fit for this app's stated design intent. Fall
back to or add Option B only if in-browser accuracy proves inadequate
for real use, since it's a meaningfully bigger lift (new server
dependencies, hundreds of MB of weights, only benefits one of the two
inference modes).
