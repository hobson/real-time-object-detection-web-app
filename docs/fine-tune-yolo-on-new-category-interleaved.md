# Fine-tuning YOLO on a new category: unfrozen + interleaved (v2)

`training/finetune_license_plate_interleaved.py` is a second strategy for
adding a new class ("license_plate") to a pretrained YOLO checkpoint,
alongside (not replacing) `training/finetune_license_plate.py`'s
frozen-weights approach. Both scripts share the same warm-start/bias-fix
machinery; they differ in how they try to avoid catastrophic forgetting of
the original 80 COCO classes while the new one is learned.

## Why this script exists

The original approach (`finetune_license_plate.py`) freezes everything
except the new class's row in the Detect head's `cv3` (classification)
branches: `freeze=21` (backbone+neck untouched) plus a per-row gradient
hook that zeros gradients for the other 80 classes on every backward pass.
This gives a strong guarantee - the other 80 classes' weights come out
byte-identical before vs. after training - but after 50 epochs on the
license-plate dataset, the `license_plate` class itself still scored
**0 precision/recall/mAP**. A fully frozen backbone/neck has no path to
learn plate-specific features (small, high-aspect-ratio objects, different
texture statistics than COCO's classes), no matter how many epochs run -
freezing solves forgetting by construction, but at the cost of also
preventing the one class that actually needs to learn something new.

This script instead:
- **Freezes nothing** (`--freeze 0` by default - trains the whole network).
- **Interleaves generic, non-plate images into every training batch** so
  the network keeps seeing ordinary, class-balanced COCO scenes and has
  continued pressure to keep detecting the original 80 classes correctly -
  protection through *exposure*, not through frozen weights.
- **Up-weights the new class's classification loss** via a `class_weights`
  hyperparameter, so recall on the (naturally rare) plate class can be
  pushed without needing to touch freezing at all.

This trades the older script's hard guarantee for a tunable
recall-vs-forgetting knob - which is the point: after the frozen approach
produced a checkpoint that could never detect its own target class, that
tradeoff needs to be explicit and adjustable rather than resolved (badly)
by construction.

## How it works

1. **Warm start, not freeze** (`_warm_start_row`, `prepare_class_head_and_weights`):
   same per-layer restoration logic as the original script (the 5-conv-layer
   `cv3` branch structure, the `attn.pe.conv` bias-reconstruction bug
   workaround, the car->license_plate row warm-start) but with no gradient
   hooks and no `BatchNorm` momentum freeze. It sets a good starting point;
   every row is free to move from there.

2. **Interleaving** (`build_interleaved_manifest`): builds a combined
   training manifest of every license-plate image plus one
   [COCO128](https://docs.ultralytics.com/datasets/detect/coco128) image
   (standard 80-class labels, no plates, ~7MB, auto-downloaded on first use
   via `ultralytics.data.utils.check_det_dataset`) inserted after every
   `--interleave-ratio` plate images. The insertion point is deterministic
   in the manifest, but ultralytics' `DataLoader` shuffles per epoch, so
   this sets the *expected* fraction of non-plate images per batch, not a
   hard per-batch guarantee - standard manifest-level class balancing, not
   a custom weighted sampler. `val.txt` is left untouched (plate-only), so
   validation mAP/recall on `license_plate` stays comparable across both
   scripts' runs.

3. **Class-weighted loss** (`--plate-class-weight`): ultralytics 8.4.105's
   `v8DetectionLoss` already supports a `model.class_weights` tensor that
   multiplies the per-class classification BCE loss
   (`ultralytics/utils/loss.py`) - this script sets it to
   `plate_class_weight` for `license_plate` and `1.0` for everything else,
   inside the same `on_train_start` callback that does the warm-start.

## Tunable hyperparameters

| Flag | Default | What it trades off |
|---|---|---|
| `--interleave-ratio` | `10` | Non-plate images per plate image (sensible range 5-20). Lower = more anti-forgetting pressure, less plate-specific training signal per epoch. |
| `--plate-class-weight` | `5.0` | Classification loss multiplier for `license_plate` (`1.0` = no boost). Higher = faster plate recall, more risk of dragging other classes' precision down. |

Both are meant to be tuned together - by hand, or by wrapping this script
in an [optuna](https://optuna.org/) study that scores trials against both
plate mAP and the other 80 classes' mAP (e.g. a weighted sum, or a
Pareto/multi-objective study) rather than optimizing either alone.

## Usage

```bash
python training/finetune_license_plate_interleaved.py \
  --epochs 50 --batch 16 \
  --interleave-ratio 10 --plate-class-weight 5.0
```

Other flags (`--base-model`, `--imgsz`, `--lr0`, `--data`, `--project`,
`--name`) match `finetune_license_plate.py`. Output goes to
`runs/license_plate/license_plate_ft_interleaved/weights/best.pt` by
default (`--name` controls the run directory, kept distinct from the
original script's `license_plate_ft` so both strategies' outputs coexist).

Generated files (regenerated fresh on every run, not meant to be
hand-edited or committed): `data/license_plates/train_interleaved.generated.txt`
(the combined manifest) and `data/license_plates/dataset_interleaved.generated.yaml`
(a copy of `dataset.yaml` with `train:` repointed at that manifest).

## COCO128/KITTI actually contain some real plates

This script's whole anti-forgetting mechanism assumes the interleaved
images are clean negatives - "definitely no plate here" - so the model
gets consistent pressure not to hallucinate plates in ordinary scenes.
That assumption doesn't fully hold: `training/label_plates_via_api.py`
(run via the `inference-server`'s `/alpr/predict` endpoint - the same
fast-alpr detector used server-side elsewhere in this repo) scanned both
pools and found real, unlabeled plates in a real minority of images:

| Dataset | Images with a real plate |
|---|---|
| COCO128 (`data/external_datasets/coco128/`) | 11 / 128 (~9%) |
| KITTI train (`data/external_datasets/kitti/`) | 813 / 5985 (~14%) |
| KITTI val | 184 / 1496 (~12%) |

Left unlabeled, every one of those images would have taught the model
"the object you just correctly detected is a false positive" - actively
fighting the plate class this whole pipeline exists to learn, not just
failing to help it.

**COCO128** (`data/external_datasets/coco128/labels/train2017/*.txt`): fixed
directly - it already shares this project's 81-class scheme (COCO's 0-79 is
what indices 0-79 mean here too), so the found plates were appended as
ordinary `80 cx cy w h` lines alongside the existing ground truth. No
further action needed; `build_interleaved_manifest` picks these up as-is.

Also worth knowing: `ultralytics`' `datasets_dir` setting was pointed at a
throwaway temp path when COCO128 was first auto-downloaded for the smoke
test below - fine for a one-off test, but it meant a fresh `check_det_dataset`
call could re-download a *clean* (unlabeled) copy at any time and silently
discard these additions. It's now repointed at
`data/external_datasets/` (`yolo settings` / `ultralytics.settings`), so
COCO128 lives at `data/external_datasets/coco128/` - stable, and consistent
with where KITTI already lives.

**KITTI** (`data/external_datasets/kitti/`) is different: it has its own
unrelated 8-class scheme (car/van/truck/pedestrian/... - see its
`kitti.yaml`), not yet remapped into this project's 81-class one, so mixing
a raw `80 ...` line into its native label files would put a plate box in
the same file as boxes indexed against a completely different class list.
The found plates instead went to a parallel
`labels_license_plate_only/{train,val}/` directory (plate-only, same
`<stem>.txt` naming) - not yet wired into any training manifest. Whoever
does the KITTI→81-class remap should merge these in at the same time,
rather than re-scanning KITTI from scratch.

Re-running the scan (e.g. after adding more interleaved datasets, or if
fast-alpr's detector improves) is idempotent - `label_plates_via_api.py`
IoU-dedupes against any plate box already in a label file, so it only ever
adds what's missing:

```bash
# from repo root, with `cd inference-server && uvicorn main:app` running separately
training/.venv/bin/python training/label_plates_via_api.py \
  --images data/external_datasets/coco128/images/train2017 \
  --labels data/external_datasets/coco128/labels/train2017 \
  --class-id 80
```

## Verified behavior

A 2-epoch smoke test (`--epochs 2 --batch 16`, CPU, taco) confirmed: COCO128
downloads and the manifest builds correctly (651 plate images + 65 COCO128
images = 716 total at the default ratio), training runs with `freeze=0` end
to end, `class_weights[license_plate]=5.0` is applied, and `license_plate`'s
mAP moves off the frozen approach's flat `0.0` (small but nonzero after just
2 epochs - full training runs are expected to need the usual tens of epochs
to reach a useful recall level, and likely some `--interleave-ratio`/
`--plate-class-weight` tuning to balance against the other classes' mAP,
which dipped noticeably in this very-short smoke test as expected from an
unfrozen backbone trained for only 2 epochs).
