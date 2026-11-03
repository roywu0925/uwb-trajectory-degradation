"""K4: same architecture and training as K3 (direct, base = anchor
position), except the velocity feature channel comes from the
position-only Kalman filter's own state instead of a finite difference
of positions (model.build_examples_k4). Same folds, same per-fold seed
convention as K1/K2/K3.

Trains all 5 folds, checkpoints each (outputs/checkpoints/K4_fold{i}.pt),
reports CV mean/spread next to K1/K2/K3's already-recorded numbers.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from cache_pipeline import build_dataset_cached
from config import OUTPUT_DIR, SPLIT_SEED, WEEKS
from kalman import build_Q, calibrate_sigma_a, estimate_sigma_a
from model import build_examples_k4
from run_tier1_taskk import build_tensors, masked_error_at_step, predict_all, train_one

PRIMARY_STEP = 19
STEP_3S = 30


def main():
    samples, _ = build_dataset_cached(WEEKS)
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)
    train_val_samples = [s for s in samples if s.game_id in train_val_games]

    sigma_a_raw = estimate_sigma_a(train_val_samples)
    sigma_a = calibrate_sigma_a(train_val_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    Q = build_Q(sigma_a)
    print(f"sigma_a = {sigma_a:.4f} (matches Task P/K/L)")

    examples = build_examples_k4(train_val_samples, Q)
    print(f"{len(examples):,} K4 examples built, anchor-frame check passed")
    tensors = build_tensors(examples)
    truth_np = tensors["truth"].numpy()
    mask_np = tensors["mask"].numpy()
    game_id_arr = tensors["game_id"]
    anchor_np = tensors["anchor"].numpy()
    base_anchor_repeat = np.repeat(anchor_np[:, None, :], truth_np.shape[1], axis=1)
    base_t = torch.from_numpy(base_anchor_repeat)

    n_folds = len(folds)
    per_fold_scores, per_fold_scores_3s = [], []
    for i in range(n_folds):
        test_games = set(folds[i])
        val_games = set(folds[(i + 1) % n_folds])
        train_games = train_val_games - test_games - val_games
        test_idx = np.where(np.isin(game_id_arr, list(test_games)))[0]
        val_idx = np.where(np.isin(game_id_arr, list(val_games)))[0]
        train_idx = np.where(np.isin(game_id_arr, list(train_games)))[0]
        seed = SPLIT_SEED + i

        model, best_val, wall_clock, epochs_run = train_one(
            "K4", base_t, tensors, train_idx, val_idx, seed, fold_id=i)
        model_out = predict_all(model, base_t, tensors, test_idx)
        pred = base_anchor_repeat[test_idx] + model_out
        err_p, n_p = masked_error_at_step(pred, truth_np[test_idx], mask_np[test_idx], PRIMARY_STEP)
        err_3, n_3 = masked_error_at_step(pred, truth_np[test_idx], mask_np[test_idx], STEP_3S)
        per_fold_scores.append(err_p)
        per_fold_scores_3s.append(err_3)
        print(f"fold {i}: epochs={epochs_run} wall_clock={wall_clock:.1f}s "
             f"@1.9s={err_p:.4f} (n={n_p}) @3s={err_3:.4f} (n={n_3})")

    s = np.array(per_fold_scores)
    s3 = np.array(per_fold_scores_3s)
    print("\n" + "=" * 70)
    print("K4 -- CV mean and fold spread")
    print("=" * 70)
    print(f"cv_mean_1.9s={s.mean():.4f} cv_std_1.9s={s.std(ddof=1):.4f} "
         f"min={s.min():.4f} max={s.max():.4f}")
    print(f"cv_mean_3s={s3.mean():.4f} cv_std_3s={s3.std(ddof=1):.4f}")

    row = pd.DataFrame([{
        "predictor": "K4", "cv_mean_1.9s": s.mean(), "cv_std_1.9s": s.std(ddof=1),
        "cv_min_1.9s": s.min(), "cv_max_1.9s": s.max(),
        "cv_mean_3s": s3.mean(), "cv_std_3s": s3.std(ddof=1),
    }]).round(4)

    cv_path = OUTPUT_DIR / "tier1_taskl_cv_summary.csv"
    existing = pd.read_csv(cv_path)
    combined = pd.concat([existing, row], ignore_index=True)
    combined.to_csv(cv_path, index=False)
    print("\nK1/K2/K3/K4 CV summary, combined:")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
