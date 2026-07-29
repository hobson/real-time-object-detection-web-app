"""Iterative class-reweighting training loop.

Alternates, per round:
  (a) train_from_db.py for --epochs-per-round epochs, with class_weights
      loaded from the previous round's evaluation (round 1 uses uniform
      weights - there is no prior checkpoint to measure fn_rate/fp_rate
      from yet, so there is nothing to base a non-uniform weight on) and
      --current-model-name set to the PREVIOUS round's name, so its
      oversampling reflects mistakes already scored for that checkpoint.
  (b) compute_class_fn_fp_rates.py against the resulting checkpoint, on the
      same combined all-categories val set, producing the weights for the
      *next* round per its Fbeta formula.
  (c) compute_label_confidence.py against the SAME resulting checkpoint
      (--current-model this round's own new best.pt, --current-model-name
      this round's own run name) - re-scores every DB label's cross_model_
      agreement against this round's actual mistakes, and persists this
      checkpoint's fresh detections, so step (a) in the *next* round has
      accurate oversampling signal instead of reusing a stale earlier
      round's. Skipped for the final round (nothing downstream would use it).

Stops when per-class weights change less than --convergence-threshold
(L-infinity norm across all classes) between two consecutive rounds, or
after --max-rounds, whichever comes first.

No freezing anywhere in this loop - both training and reweighting act on
the whole network's classification head, per the same class_weights
mechanism train_from_db.py and finetune_license_plate_interleaved.py
already use (ultralytics' own v8DetectionLoss.class_weights support - see
finetune_license_plate_interleaved.py's module docstring).

Every child script is invoked as a subprocess (not imported and called
directly) since each is its own argparse-driven entry point with process-
global state (generated manifests/dataset yamls written to fixed paths) -
running them out-of-process avoids one round's generated files leaking into
the next via stale module-level state.

Usage: python training/iterative_reweight_train.py \
    [--checkpoint runs/detect/runs/license_plate/license_plate_ft_reweight_round3/weights/best.pt] \
    [--initial-model-name license_plate_ft_reweight_round3] \
    [--max-rounds 3] [--epochs-per-round 10] [--convergence-threshold 0.1]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import torch
import yaml

TRAINING_DIR = Path(__file__).parent
REPO_ROOT = TRAINING_DIR.parent
DATASET_DIR = REPO_ROOT / "data" / "license_plates"
EVAL_DATA_YAML = REPO_ROOT / "data" / "external_datasets" / "all_categories_dataset.generated.yaml"
WEIGHTS_DIR = TRAINING_DIR / "reweight_rounds"


def run(cmd: list[str]) -> None:
    print(f"[iterative-reweight] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="runs/detect/runs/license_plate/license_plate_ft_reweight_round3/weights/best.pt",
        help="Starting checkpoint - a previously fine-tuned 81(+)-class model, not the original pretrained model.",
    )
    parser.add_argument(
        "--initial-model-name", default="license_plate_ft_reweight_round3",
        help="model_name --checkpoint's detections were persisted under by the most recent "
        "compute_label_confidence.py run against it - used as round 1's --current-model-name.",
    )
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument(
        "--epochs-per-round", type=int, default=10,
        help="Passed to train_from_db.py's --epochs - see its help for why the default was lowered "
        "from 50: shorter rounds mean more frequent round-boundary resume points (see "
        "--skip-training-for-round) on top of train_from_db.py's own per-epoch F1-best checkpointing.",
    )
    parser.add_argument("--convergence-threshold", type=float, default=0.1)
    parser.add_argument("--agreement-threshold", type=float, default=0.67, help="Passed to train_from_db.py - see its --agreement-threshold help.")
    parser.add_argument("--oversample-repeats", type=int, default=3, help="Passed to train_from_db.py.")
    parser.add_argument("--beta", type=float, default=2.0, help="Passed to compute_class_fn_fp_rates.py - see its --beta help.")
    parser.add_argument("--min-weight", type=float, default=0.1)
    parser.add_argument("--max-weight", type=float, default=10.0)
    parser.add_argument("--project", default="runs/license_plate")
    parser.add_argument(
        "--device", default="cpu",
        help="Passed to train_from_db.py, compute_class_fn_fp_rates.py, and compute_label_confidence.py "
        "- each verifies it independently via device_check.resolve_device before use. See train_from_db.py's "
        "--device help.",
    )
    parser.add_argument("--cos-lr", action="store_true", help="Passed to train_from_db.py - see its --cos-lr help.")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Passed to train_from_db.py - see its --warmup-epochs help.")
    parser.add_argument("--multi-scale", action="store_true", help="Passed to train_from_db.py - see its --multi-scale help.")
    parser.add_argument(
        "--skip-training-for-round", type=int, default=None,
        help="If set, skip train_from_db.py for this one round number and read its checkpoint "
        "from the existing from_db_round{N}_best_path.txt instead (which must already exist) - "
        "for resuming a run where that round's training already finished but the trailing "
        "fn/fp-rate + confidence-rescoring steps never ran (e.g. after an SSH disconnect killed "
        "the parent process mid-loop). --checkpoint/--initial-model-name are unused for this "
        "round when set, since its real checkpoint/name come from the existing best_path_out file.",
    )
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_config = yaml.safe_load((DATASET_DIR / "dataset.yaml").read_text())
    num_classes = len(dataset_config["names"])

    # "from_db" prefix throughout (run names, weight files) - distinct from
    # the earlier file-based round1/round2/round3 runs this session already
    # produced, so this loop can't collide with or overwrite them.
    checkpoint = args.checkpoint
    current_model_name = args.initial_model_name
    prev_weights = torch.ones(num_classes)
    weights_path = WEIGHTS_DIR / "from_db_round0_uniform.pt"
    torch.save(prev_weights, weights_path)

    for round_i in range(1, args.max_rounds + 1):
        run_name = f"license_plate_ft_from_db_round{round_i}"
        best_path_out = WEIGHTS_DIR / f"from_db_round{round_i}_best_path.txt"

        if round_i == args.skip_training_for_round:
            if not best_path_out.exists():
                raise SystemExit(
                    f"--skip-training-for-round {round_i} but {best_path_out} doesn't exist - "
                    "that round's training hasn't actually finished, nothing to resume from."
                )
            checkpoint = best_path_out.read_text().strip()
            current_model_name = run_name
            print(f"[iterative-reweight] Round {round_i}: skipping training (resuming), using existing checkpoint {checkpoint}")
        else:
            train_cmd = [
                sys.executable, "training/train_from_db.py",
                "--checkpoint", checkpoint,
                "--current-model-name", current_model_name,
                "--epochs", str(args.epochs_per_round),
                "--agreement-threshold", str(args.agreement_threshold),
                "--oversample-repeats", str(args.oversample_repeats),
                "--class-weights-file", str(weights_path),
                "--project", args.project,
                "--name", run_name,
                "--best-path-out", str(best_path_out),
                "--device", args.device,
                "--warmup-epochs", str(args.warmup_epochs),
            ]
            train_cmd += [f"--{flag}" for flag, enabled in [("cos-lr", args.cos_lr), ("multi-scale", args.multi_scale)] if enabled]
            run(train_cmd)
            checkpoint = best_path_out.read_text().strip()
            current_model_name = run_name
            print(f"[iterative-reweight] Round {round_i} checkpoint: {checkpoint}")

        new_weights_path = WEIGHTS_DIR / f"from_db_round{round_i}.pt"
        run([
            sys.executable, "training/compute_class_fn_fp_rates.py",
            "--model", checkpoint,
            "--data", str(EVAL_DATA_YAML),
            "--beta", str(args.beta),
            "--min-weight", str(args.min_weight),
            "--max-weight", str(args.max_weight),
            "--out", str(new_weights_path),
            "--device", args.device,
        ])

        new_weights = torch.load(new_weights_path)
        delta = (new_weights - prev_weights).abs().max().item()
        print(f"[iterative-reweight] Round {round_i}: max per-class weight change vs previous round = {delta:.4f}")
        converged = delta < args.convergence_threshold
        if converged:
            print(f"[iterative-reweight] Converged (< {args.convergence_threshold}) after round {round_i} - stopping")

        prev_weights = new_weights
        weights_path = new_weights_path

        if converged or round_i == args.max_rounds:
            break

        # Re-score THIS round's own checkpoint (not the next round's, which
        # doesn't exist yet) so train_from_db.py's oversampling in the next
        # iteration reflects mistakes this checkpoint actually makes, rather
        # than reusing current_model_name's now-stale prior-round scoring.
        run([
            sys.executable, "training/compute_label_confidence.py",
            "--current-model", checkpoint,
            "--current-model-name", current_model_name,
            "--device", args.device,
        ])

    print(f"[iterative-reweight] Final checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
