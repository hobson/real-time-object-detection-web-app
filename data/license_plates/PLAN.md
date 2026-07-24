# License plate detection: dataset + fine-tuning plan

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

**Layout** (this directory):
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

7. **Wire it into the app** (both inference paths use the same output
   format, so both need the same three edits):
   - Drop the new `.onnx` into `/models/`.
   - Add a `RES_TO_MODEL` entry in `components/models/Yolo.tsx`
     (client-side) and the equivalent `RES_TO_MODEL`/model list in
     `inference-server/main.py` (server-side).
   - Append `"license_plate"` to `data/yolo_classes.ts` *and*
     `inference-server/yolo_classes.py` (kept in sync manually - see the
     comment already in the Python file), and bump the `80` in
     `NUM_CLASSES`/`4 + NUM_CLASSES` slicing in both
     `postprocessYolov11`/`postprocessYolov12` (`Yolo.tsx`) and
     `postprocess_yolov11_12` (`inference-server/postprocess.py`) to `81`.
   - Add `"license_plate"` to `NOTIFY_CLASSES` in `Yolo.tsx` /
     `YoloServer.tsx` if plate detections should also trigger the ntfy
     notification (currently only `person`/`car` do).

### Open question worth deciding before starting

Whether to fine-tune `yolo12n` (this app's current default, opset 19,
needs `onnxruntime-web >= 1.18` per the earlier debugging this session) or
`yolov7-tiny` (older opset, most battle-tested in this app's actual
deployment history). No strong reason to prefer one over the other for
this specific task — pick whichever this app ends up standardizing on
long-term.
