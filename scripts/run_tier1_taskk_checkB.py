"""One-off diagnostic: fold 0, K2 only. Not part of the Task K pipeline.

1. Trains K2 on fold 0 (checkpointed via run_tier1_taskk.save_checkpoint,
   which now runs on every train_one call permanently).
2. Fills the missing "K2 model, mean over unmasked steps 1..h" cell from
   the user's Check A table.
3. Alignment check: for every fold-0 track with horizon >= 19, find which
   truth column the model's (and, as a control, velocity_turning_to_ball's)
   step-19 prediction is actually closest to. A sharp peak at column 18
   (0-indexed) means the arrays are aligned and step 19 means step 19 for
   both. No scoring code (model.py, run_tier1_taskk.masked_error_at_step)
   is touched.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from cache_pipeline import build_dataset_cached
from config import MAX_HORIZON, OUTPUT_DIR, SPLIT_SEED, WEEKS
from kalman import calibrate_sigma_a, estimate_sigma_a
from model import build_examples
from run_tier1_taskk import (build_tensors, compute_base_arrays,
                             masked_error_at_step, predict_all, train_one)
from run_tier1_taskp import build_predictors_quadrature

PRIMARY_STEP = 19
PRIMARY_COL = PRIMARY_STEP - 1


def main():
    samples, _ = build_dataset_cached(WEEKS)
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)
    train_val_samples = [s for s in samples if s.game_id in train_val_games]

    examples = build_examples(train_val_samples)
    tensors = build_tensors(examples)
    truth_np = tensors["truth"].numpy()
    mask_np = tensors["mask"].numpy()
    game_id_arr = tensors["game_id"]

    sigma_a_raw = estimate_sigma_a(train_val_samples)
    sigma_a = calibrate_sigma_a(train_val_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    predictors = build_predictors_quadrature(sigma_a)
    base_vtb_np = compute_base_arrays(train_val_samples, predictors["velocity_turning_to_ball"], examples)

    i = 0
    test_games = set(folds[i])
    val_games = set(folds[(i + 1) % len(folds)])
    train_games = train_val_games - test_games - val_games
    test_idx = np.where(np.isin(game_id_arr, list(test_games)))[0]
    val_idx = np.where(np.isin(game_id_arr, list(val_games)))[0]
    train_idx = np.where(np.isin(game_id_arr, list(train_games)))[0]

    base_vtb = torch.from_numpy(base_vtb_np)
    seed = SPLIT_SEED + i

    # -------- 1. train + checkpoint (permanent side effect, outputs/checkpoints/)
    model, best_val, wall_clock, epochs_run = train_one(
        "K2", base_vtb, tensors, train_idx, val_idx, seed, fold_id=i, checkpoint=True)
    print(f"trained K2 fold0: val_loss={best_val:.4f} epochs={epochs_run} wall_clock={wall_clock:.1f}s")

    model_out = predict_all(model, base_vtb, tensors, test_idx)
    pred = base_vtb[test_idx].numpy() + model_out
    truth = truth_np[test_idx]
    mask = mask_np[test_idx]

    # -------- 2. fill the missing cell: (i) mean over unmasked steps 1..h
    err_ii, n_ii = masked_error_at_step(pred, truth, mask, PRIMARY_STEP)
    dist_all = np.linalg.norm(pred - truth, axis=-1)
    per_track_mean = (dist_all * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
    err_i = per_track_mean.mean()

    print("\n" + "=" * 70)
    print("Four-cell table, fold 0 out-of-fold, completed")
    print("=" * 70)
    print(f"{'predictor':28s} | (i) mean over steps | (ii) at step h")
    base_err_ii, base_n_ii = masked_error_at_step(base_vtb_np[test_idx], truth, mask, PRIMARY_STEP)
    base_dist_all = np.linalg.norm(base_vtb_np[test_idx] - truth, axis=-1)
    base_per_track_mean = (base_dist_all * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
    base_err_i = base_per_track_mean.mean()
    print(f"{'velocity_turning_to_ball':28s} | {base_err_i:20.4f} | {base_err_ii:14.4f}")
    print(f"{'K2 model':28s} | {err_i:20.4f} | {err_ii:14.4f}  (n={n_ii})")

    # -------- 3. alignment check
    valid19 = mask[:, PRIMARY_COL] > 0
    print(f"\nfold-0 tracks with horizon >= 19: {valid19.sum()}")

    def argmin_histogram(pred_arr: np.ndarray, label: str):
        pred_at_19 = pred_arr[valid19, PRIMARY_COL]          # (n, 2)
        truth_sub = truth[valid19]                            # (n, MAX_HORIZON, 2)
        mask_sub = mask[valid19]                               # (n, MAX_HORIZON)
        d = np.linalg.norm(truth_sub - pred_at_19[:, None, :], axis=-1)  # (n, MAX_HORIZON)
        d_masked = np.where(mask_sub > 0, d, np.inf)
        argmin_col = d_masked.argmin(axis=1)                   # 0-indexed column
        hist = np.bincount(argmin_col, minlength=MAX_HORIZON)
        print(f"\n{label}: argmin column histogram (0-indexed; 18 == step 19 == 1.9s)")
        for c in range(MAX_HORIZON):
            if hist[c] > 0:
                bar = "#" * int(hist[c])
                marker = "  <-- step 19" if c == PRIMARY_COL else ""
                print(f"  col {c:2d} (step {c+1:2d}, {(c+1)/10:.1f}s): {hist[c]:4d} {bar}{marker}")
        print(f"  mode: col {int(np.argmax(hist))} (step {int(np.argmax(hist))+1})")
        print(f"  median: col {int(np.median(argmin_col))} (step {int(np.median(argmin_col))+1})")
        print(f"  fraction with argmin exactly at col {PRIMARY_COL} (step 19): "
             f"{(argmin_col == PRIMARY_COL).mean():.3f}")
        return argmin_col

    argmin_model = argmin_histogram(pred, "K2 model")
    argmin_base = argmin_histogram(base_vtb_np[test_idx], "velocity_turning_to_ball (control)")


if __name__ == "__main__":
    main()
