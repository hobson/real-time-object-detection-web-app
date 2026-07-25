"""Fine-tune yolo12n to add a "license_plate" 81st class, warm-started from
"car" (COCO index 2), per docs/PLAN-realtime-license-plate-detection.md
sections 2-3.

Usage: python training/finetune_license_plate.py [--epochs 50] [--imgsz 256]

Each cv3 (classification) branch has 5 actual conv layers, not 3 - an
earlier version of this script only found/protected 3 of them (a bug,
found by noticing the exported model's raw output was ~100% license_plate
regardless of input - the other 80 classes' shared feature pathway had
been silently corrupted by 50 epochs of unprotected training, even though
the final per-class weight matrix itself was correctly frozen). The full
structure, per branch:

  branch[0] = Sequential(DWConv(64->64, groups=64), Conv(64->nc))
  branch[1] = Sequential(DWConv(nc->nc, groups=nc), Conv(nc->nc))
  branch[2] = Conv2d(nc->nc)  # bare, has bias; the other 4 don't

Three different protection strategies are needed:
- branch[0][0]: channel count is fixed (64, doesn't depend on nc) and
  there's no per-class structure to preserve selectively - just freeze
  it completely (no gradient at all).
- branch[1][0]: depthwise (groups=nc), so channel i's output depends
  ONLY on input channel i - no cross-channel mixing, so no "new input
  column" concern. Preserve channel-wise like a normal nc-indexed layer.
- branch[0][1], branch[1][1], branch[2]: regular (non-depthwise) convs
  where output width is nc-dependent. When the *input* width is also
  nc-dependent (i.e. it receives another nc-width layer's output),
  explicitly zero the old classes' weights for the new input column(s) -
  otherwise the new class's pathway leaks into old classes' output even
  with their own weight rows otherwise byte-identical to pretrained.

For every layer: restore rows [0:80] from the true pretrained checkpoint,
warm-start row 80 (license_plate) from car's (index 2) weights scaled
down + confidence-biased low, register a gradient hook zeroing rows
[0:80] every backward pass so they can't drift during training, and (for
layers with BatchNorm) freeze the running stats (momentum=0) - weight/
bias hooks alone don't stop running_mean/running_var drifting via
forward-pass EMA updates in train mode.

Combined with freeze=21 (backbone/neck stay untouched; only the Detect
head trains at all), this should make the other 80 classes' detection
head byte-identical to pretrained before vs. after training - verify
with the same exact-comparison approach used to find this bug (compare
every cv3 sub-layer, not just the final one, weight+bias+BN running
stats, torch.equal not allclose) before trusting a training run's output.
"""
import argparse

import torch
from ultralytics import YOLO

CAR_CLASS_IDX = 2  # COCO index for "car" - the warm-start source
NEW_CLASS_IDX = 80  # license_plate becomes the 81st class (index 80)
NUM_ORIGINAL_CLASSES = 80


def _cv3_layers(branch):
    """All 5 actual conv layers in one cv3 branch, tagged with how each
    needs to be handled. Returns [(conv, bn_or_None, kind), ...].
    """
    return [
        (branch[0][0].conv, branch[0][0].bn, "freeze"),
        (branch[0][1].conv, branch[0][1].bn, "regular"),
        (branch[1][0].conv, branch[1][0].bn, "depthwise"),
        (branch[1][1].conv, branch[1][1].bn, "regular"),
        (branch[2], None, "regular"),
    ]


def _freeze_completely(l_conv, l_bn):
    l_conv.weight.requires_grad_(False)
    if l_conv.bias is not None:
        l_conv.bias.requires_grad_(False)
    if l_bn is not None:
        l_bn.weight.requires_grad_(False)
        l_bn.bias.requires_grad_(False)
        l_bn.momentum = 0.0


def _restore_and_guard(p_conv, p_bn, l_conv, l_bn, kind):
    with torch.no_grad():
        if kind == "depthwise":
            # groups=channels: output channel i depends only on input
            # channel i, so there's no "new input column" to worry about -
            # a straight per-channel copy is exact.
            l_conv.weight[:NUM_ORIGINAL_CLASSES] = p_conv.weight
            l_conv.weight[NEW_CLASS_IDX] = p_conv.weight[CAR_CLASS_IDX] * 0.5
        else:
            in_old = p_conv.weight.shape[1]
            in_new = l_conv.weight.shape[1]
            l_conv.weight[:NUM_ORIGINAL_CLASSES, :in_old] = p_conv.weight
            l_conv.weight[NEW_CLASS_IDX, :in_old] = p_conv.weight[CAR_CLASS_IDX] * 0.5
            if in_new > in_old:
                # New input column(s) come from another nc-width layer's
                # new output channel - old classes must get literally zero
                # contribution from it, or the new class's pathway leaks
                # into their output even with everything else preserved.
                l_conv.weight[:NUM_ORIGINAL_CLASSES, in_old:] = 0
                l_conv.weight[NEW_CLASS_IDX, in_old:] = 0

        if l_conv.bias is not None:
            l_conv.bias[:NUM_ORIGINAL_CLASSES] = p_conv.bias
            l_conv.bias[NEW_CLASS_IDX] = p_conv.bias[CAR_CLASS_IDX] - 2.0

        if l_bn is not None and p_bn is not None:
            l_bn.weight[:NUM_ORIGINAL_CLASSES] = p_bn.weight
            l_bn.bias[:NUM_ORIGINAL_CLASSES] = p_bn.bias
            l_bn.running_mean[:NUM_ORIGINAL_CLASSES] = p_bn.running_mean
            l_bn.running_var[:NUM_ORIGINAL_CLASSES] = p_bn.running_var
            l_bn.weight[NEW_CLASS_IDX] = p_bn.weight[CAR_CLASS_IDX]
            l_bn.bias[NEW_CLASS_IDX] = p_bn.bias[CAR_CLASS_IDX]
            l_bn.running_mean[NEW_CLASS_IDX] = p_bn.running_mean[CAR_CLASS_IDX]
            l_bn.running_var[NEW_CLASS_IDX] = p_bn.running_var[CAR_CLASS_IDX]
            l_bn.momentum = 0.0  # freeze running stats entirely

    def zero_old_class_grad(grad, n=NUM_ORIGINAL_CLASSES):
        grad = grad.clone()
        grad[:n] = 0
        return grad

    l_conv.weight.register_hook(zero_old_class_grad)
    if l_conv.bias is not None:
        l_conv.bias.register_hook(zero_old_class_grad)
    if l_bn is not None:
        l_bn.weight.register_hook(zero_old_class_grad)
        l_bn.bias.register_hook(zero_old_class_grad)


ATTENTION_BLOCK_INDICES = [6, 8]  # backbone A2C2f blocks (yolo12n specifically)


def restore_attention_position_encoding_bias(live_model, base_model_path: str):
    """Works around an ultralytics version-skew bug (found empirically,
    not documented anywhere): the currently-installed ultralytics
    reconstructs A2C2f's attn.pe.conv layers with bias=False, but the
    actual released yolo12n.pt checkpoint has bias=True there. This has
    nothing to do with nc/class-count changes - it reproduces identically
    for a from-scratch nc=80 reconstruction too - it's purely an
    architecture-definition mismatch between whatever ultralytics version
    originally exported this checkpoint and the one installed now.

    Silently dropping 8 small bias terms (4 per attention block x 2
    blocks) sounds minor but measurably degraded every class's output in
    testing (e.g. car's raw confidence on a test frame dropped from an
    exact-baseline-matching 0.378 to 0.074, ~5x) - YOLOv12's A2C2f blocks
    lean heavily on attention, so even small positional-encoding bias
    differences shift attention patterns more than a typical small-weight
    perturbation would. Confirmed fixed: restoring these 8 biases makes
    old classes' raw output byte-identical to the untouched pretrained
    model (not just close) in testing.

    This is NOT part of the class-head surgery (prepare_class_head) since
    it's unrelated to adding a class - it's a general reconstruction-
    fidelity bug that would apply to ANY fine-tune of this checkpoint
    that changes nc, even without adding a new class at all.
    """
    pretrained = YOLO(base_model_path)
    restored = 0
    for module_idx in ATTENTION_BLOCK_INDICES:
        pretrained_block = pretrained.model.model[module_idx]
        live_block = live_model.model[module_idx]
        for i in range(len(pretrained_block.m)):
            for j in range(len(pretrained_block.m[i])):
                p_conv = pretrained_block.m[i][j].attn.pe.conv
                l_conv = live_block.m[i][j].attn.pe.conv
                if p_conv.bias is not None and l_conv.bias is None:
                    l_conv.bias = torch.nn.Parameter(p_conv.bias.clone())
                    l_conv.bias.requires_grad_(False)  # backbone is frozen anyway (freeze=21)
                    restored += 1
    print(f"[finetune] Restored {restored} attn.pe.conv bias terms dropped by reconstruction")


def prepare_class_head(trainer, base_model_path: str):
    """Runs once, at on_train_start - trainer.model is by this point
    ultralytics' own nc=81 reconstruction of the Detect head. This
    explicitly restores/warm-starts/guards every cv3 sub-layer rather
    than trusting that reconstruction, then locks the old 80 classes
    down for the rest of training. Also fixes an unrelated backbone
    reconstruction bug - see restore_attention_position_encoding_bias.

    IMPORTANT: the attn.pe bias fix applied here does NOT survive
    ultralytics' end-of-training checkpoint stripping (confirmed
    empirically - it's gone again in the saved best.pt/last.pt, likely
    because strip_optimizer() re-derives the module rather than deep-
    copying the live one as-is). It still needs to be re-applied as a
    post-processing step on the final saved checkpoint - see main().
    Applying it here too is still worthwhile: it keeps validation
    metrics computed *during* training meaningful (they'd otherwise
    reflect the degraded, bias-missing backbone).
    """
    restore_attention_position_encoding_bias(trainer.model, base_model_path)

    pretrained = YOLO(base_model_path)
    pretrained_head = pretrained.model.model[-1]
    live_head = trainer.model.model[-1]
    assert live_head.nc == NUM_ORIGINAL_CLASSES + 1, f"expected nc=81, got {live_head.nc}"

    for pretrained_branch, live_branch in zip(pretrained_head.cv3, live_head.cv3):
        pretrained_layers = _cv3_layers(pretrained_branch)
        live_layers = _cv3_layers(live_branch)

        for (p_conv, p_bn, _), (l_conv, l_bn, kind) in zip(
            pretrained_layers, live_layers
        ):
            if kind == "freeze":
                _freeze_completely(l_conv, l_bn)
            else:
                _restore_and_guard(p_conv, p_bn, l_conv, l_bn, kind)

    print(
        f"[finetune] Restored + guarded {len(pretrained_head.cv3)} cv3 branches "
        f"x 5 layers each (weight/bias/BN running stats)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="yolo12n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--freeze", type=int, default=21)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--data", default="data/license_plates/dataset.yaml")
    parser.add_argument("--project", default="runs/license_plate")
    args = parser.parse_args()

    print(f"[finetune] Loading base model {args.base_model}")
    model = YOLO(args.base_model)
    # Metadata only - just tells ultralytics there's an 81st class named
    # license_plate, so it reconstructs the head at the right width and
    # attempts its own by-name remap for the other 80. The actual weight
    # correctness is enforced in prepare_class_head() below regardless of
    # how well that remap does on its own.
    names = dict(model.model.names)
    names[NEW_CLASS_IDX] = "license_plate"
    model.model.names = names
    model.model.yaml["nc"] = NUM_ORIGINAL_CLASSES + 1

    seed_path = "training/yolo12n_81class_seed.pt"
    model.save(seed_path)
    print(f"[finetune] Saved seed checkpoint to {seed_path}")

    train_model = YOLO(seed_path)
    train_model.add_callback(
        "on_train_start", lambda trainer: prepare_class_head(trainer, args.base_model)
    )

    print(
        f"[finetune] Starting training: epochs={args.epochs} imgsz={args.imgsz} freeze={args.freeze}"
    )
    train_model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        lr0=args.lr0,
        project=args.project,
        name="license_plate_ft",
        exist_ok=True,
    )

    # Re-apply the attn.pe bias fix to the actual saved output - see the
    # docstring on prepare_class_head for why this doesn't survive
    # ultralytics' own checkpoint-saving process and must be redone here.
    # (Ask the trainer for its own save_dir rather than reconstructing the
    # path by hand - ultralytics silently prepends a task subdir, e.g.
    # "detect/", that --project alone doesn't fully control.)
    best_path = str(train_model.trainer.save_dir / "weights" / "best.pt")
    print(f"[finetune] Re-applying attn.pe bias fix to final checkpoint {best_path}")
    final_model = YOLO(best_path)
    restore_attention_position_encoding_bias(final_model.model, args.base_model)
    final_model.save(best_path)
    print(f"[finetune] Done - {best_path} is ready to export")


if __name__ == "__main__":
    main()
