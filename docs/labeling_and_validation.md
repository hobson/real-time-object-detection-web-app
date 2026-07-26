# Labeling new footage + fine-tuning/validating the YOLO plate detector

This covers turning raw clips (e.g. `data/unlabeled/*.mp4`) into labeled
training data for the custom in-browser `license_plate` class described in
[`PLAN-realtime-license-plate-detection.md`](./PLAN-realtime-license-plate-detection.md),
and how to fine-tune/validate against it. It's a different model from
`fast-alpr`'s own pretrained YOLOv9 detector (used server-side, see the main
`CLAUDE.md`'s "License plate detection + OCR" section) — `fast-alpr` doesn't
ship a training pipeline of its own; this whole doc only applies to the
custom `yolo12n` + `license_plate`-as-81st-class model trained via
`training/finetune_license_plate.py`.

## 1. Extract frames to label

Pick a handful of frames per video rather than every frame — adjacent video
frames are near-duplicates and add little. `ffmpeg` at ~1 fps is enough:

```bash
mkdir -p /tmp/label_frames
ffmpeg -i data/unlabeled/pexels-4608285-us-highway-dashcam.mp4 \
  -vf fps=1 /tmp/label_frames/highway_%03d.jpg
```

Skip frames with no plate visible at all (checked earlier: the highway and
ambulance clips in `data/unlabeled/` have no in-frame plates at native
camera angle/distance — see `SOURCES.md` — so labeling those specific clips
as-is would only add background/negative examples, not positive plate
boxes; still useful for reducing false positives, just not for teaching the
model what a plate looks like).

## 2. Label the plates (draw boxes → YOLO format)

**Recommended tool: [`labelImg`](https://github.com/HumanSignal/labelImg)**
— simplest option, works fully offline, writes YOLO format directly.

```bash
uv run --with labelImg labelImg /tmp/label_frames
```

In labelImg: `View → Auto Save mode`, set format to **YOLO** (button
cycles PascalVOC/YOLO/CreateML), then for each image draw a box per plate
(`W` to start a box) and pick/create the `license_plate` class.

Alternatives if you want browser-based or multi-person labeling:
- **[CVAT](https://github.com/opencv/cvat)** (self-hosted via `docker
  compose`, or cvat.ai) — better for larger batches or a team, exports to
  YOLO format.
- **[Roboflow](https://roboflow.com)** — free tier, browser-based, also
  exports YOLO format; needs an account (this is why the original dataset
  used Open Images instead — see `PLAN-realtime-license-plate-detection.md`
  §1 — but for a handful of new images an account is a non-issue).
- **`training/label_app.py`** (this repo, Streamlit) — a first-cut in-house
  tool, useful specifically because it captures the plate's text and
  issuing state alongside the box in one pass, which none of the tools
  above do (YOLO format has no field for that, so it writes them to a
  separate CSV). Numeric text fields for `cx cy w h` rather than a
  drag-to-draw canvas — faster to build, slower to use for many images than
  labelImg/CVAT's click-and-drag. Run it:

  ```bash
  cd training && uv run --extra label streamlit run label_app.py
  ```

  Defaults to labeling `data/license_plates/images` into
  `data/license_plates/labels_plates_only/` (YOLO boxes, class `80`) plus
  `data/license_plates/plate_metadata.csv` (`stem,plate_text,state`); pass
  `--images`/`--labels`/`--metadata` to point at a different directory
  (e.g. frames extracted from `data/unlabeled/` per §1 above). The image
  preview redraws the box live as you edit the four numbers, which is the
  main check against typos before saving.
- **`training/label_plates_via_api.py`** — not hand-labeling at all: calls
  the running `inference-server`'s `/alpr/predict` endpoint (fast-alpr's own
  plate detector) to auto-label plates it finds, IoU-deduped against
  anything already in the label file so it's safe to re-run. Useful for
  bulk-scanning a large pool of images no one has looked at yet (e.g.
  confirming a "no plates here" dataset actually has none - see
  [`fine-tune-yolo-on-new-category-interleaved.md`](./fine-tune-yolo-on-new-category-interleaved.md)'s
  "COCO128/KITTI actually contain some real plates" section for why that
  check matters and what it found). Treat its output as a starting point to
  spot-check, not ground truth — it's the same model this whole pipeline is
  trying to fine-tune past, so it will share that model's blind spots.

### YOLO label format (what these tools write, and what to hand-edit if needed)

One `.txt` file per image, same stem as the image, one line per box:

```
<class_id> <cx> <cy> <w> <h>
```

- `cx, cy, w, h` are box center/width/height, **normalized 0–1** by image
  width/height (not pixels).
- `class_id` must be **`80`** to match this project's `dataset.yaml` (COCO's
  80 classes, 0–79, plus `license_plate` appended at index 80 — see the
  `names:` block in `data/license_plates/dataset.yaml`). Using `0` (a fresh
  single-class scheme) will silently corrupt training if mixed into this
  dataset — every image's plate would get labeled as `person`.
- An image with no plate gets an empty (zero-byte) label file, not a
  missing one — YOLO treats missing/empty the same (negative example), but
  an accidentally-missing file next to an image ultralytics can't decide is
  intentional or not.
- To hand-edit a box (e.g. nudge coordinates after auto-labeling), any text
  editor works — the four numbers are self-explanatory once you know the
  normalization; there's no benefit to round-tripping through a GUI just to
  tweak numbers.

## 3. Add the new labels into the dataset

```
data/license_plates/
├── images/               # drop new .jpg here
├── labels/                # matching .txt here (class 80 + any pseudo-labels)
├── labels_plates_only/    # matching .txt here too — untouched, plate-only ground truth
├── train.txt              # add new image paths (90/10 split with val.txt)
└── val.txt
```

`labels_plates_only/` is the ground-truth plate annotations before
`training/pseudo_label_plates.py` adds other-class (car/person/etc.) boxes
predicted by a pretrained model — see that script's docstring for why (the
other 80 classes are real and visible in these images but unlabeled, which
would otherwise teach the model to treat them as false positives). Put your
new hand-labels in `labels_plates_only/` (plate boxes only, class `80`),
then regenerate `labels/`:

```bash
training/.venv/bin/python training/pseudo_label_plates.py
```

(run from the repo root — that script's own default `--data`-style paths are
repo-root-relative, and `training/.venv/bin/python` runs it with
`training/`'s dependencies without needing `uv run` to resolve a project;
`uv run` looks for a `pyproject.toml` in the *current* directory, which only
exists in `training/`, not the repo root)

Add new image paths to `train.txt` (most) or `val.txt` (a held-out
fraction, roughly the existing ~90/10 split) — paths are relative to
`data/license_plates/`, e.g. `./images/highway_003.jpg`.

## 4. Fine-tune

```bash
training/.venv/bin/python training/finetune_license_plate.py --epochs 50 --imgsz 256
```

Reads `data/license_plates/dataset.yaml` (path is hardcoded absolute in
that file — see the comment there on why). Output checkpoint:
`runs/license_plate/license_plate_ft/weights/best.pt`. See the extensive
module docstring in `finetune_license_plate.py` for what this script
actually does (warm-starts the new class from `car`'s weights, freezes the
other 80 classes so they can't drift, freezes the backbone — only the
detection head trains).

## 5. Validate — get accuracy statistics

Ultralytics' own `val` mode computes precision/recall/mAP against
`val.txt` directly from a checkpoint, no separate script needed:

```bash
training/.venv/bin/yolo detect val \
  model=runs/license_plate/license_plate_ft/weights/best.pt \
  data=data/license_plates/dataset.yaml
```

(again from the repo root — `runs/` here means repo-root `runs/`, which is
where step 4's `training/.venv/bin/python training/finetune_license_plate.py`
invocation above writes its output, since it also runs from the repo root.
This checkpoint won't exist until you've actually run step 4 to completion -
if you just want to smoke-test this command's plumbing before a real
training run finishes, point `model=` at the untrained base
`yolo12n.pt`/`yolo11n.pt` in the repo root instead, understanding that its
`license_plate` numbers will be meaningless since it was never trained on
that class.)

This prints, per class (including `license_plate` specifically) and
overall:
- **Precision / Recall**
- **mAP50** (mean average precision at IoU≥0.5 — the usual headline number)
- **mAP50-95** (averaged over IoU thresholds 0.5–0.95, stricter)

It also saves a confusion matrix and PR-curve plots to
`runs/license_plate/<val-run>/`. To check *only* the `license_plate` class's
numbers (ignore the other 80 COCO classes' scores, which shouldn't have
moved — see the fine-tuning script's freezing logic), look at the
`license_plate` row specifically rather than the "all classes" summary row.

To sanity-check a specific image/video qualitatively (not just aggregate
stats), `yolo detect predict` overlays boxes you can eyeball:

```bash
training/.venv/bin/yolo detect predict \
  model=runs/license_plate/license_plate_ft/weights/best.pt \
  source=data/unlabeled/pexels-4608285-us-highway-dashcam.mp4 \
  conf=0.25
```
