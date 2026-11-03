"""Tier 1 Task L (TIER1_SPEC_6.md, amending TIER1_SPEC_4.md): the CV
portion. All 5 folds, all 3 model variants (K1, K2, K3), the FULL
baseline set (13 predictors, not just the 4 in the Task P headline)
recomputed out-of-fold on identical plays, per-role breakdown, CV mean
and fold spread for every model row.

Test split is NOT touched here -- that is a separate, explicitly
confirmed step per the project's standing rule.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from cache_pipeline import build_dataset_cached
from config import OUTPUT_DIR, SPLIT_SEED, WEEKS
from kalman import calibrate_sigma_a, estimate_sigma_a
from model import build_examples
from run_tier1_taskk import (build_tensors, compute_base_arrays,
                             masked_error_at_step, predict_all, train_one)
from run_tier1_taskp import build_predictors_quadrature

PRIMARY_STEP = 19
STEP_3S = 30

ALL_BASELINE_NAMES = [
    "constant_position", "constant_velocity", "constant_acceleration",
    "damped_velocity", "toward_ball_land", "velocity_turning_to_ball",
    "kalman", "kalman_velocity", "kalman_ca", "imm",
    "tbl_kalman_speed", "tbl_kalman_pos", "tbl_kalman_both",
]


def masked_error_by_role(pred, truth, mask, roles, step):
    col = step - 1
    valid = mask[:, col] > 0
    if valid.sum() == 0:
        return {}
    err = np.linalg.norm(pred[valid, col] - truth[valid, col], axis=-1)
    r = roles[valid]
    out = {}
    for role in np.unique(r):
        m = r == role
        out[role] = (float(err[m].mean()), int(m.sum()))
    out["overall"] = (float(err.mean()), int(valid.sum()))
    return out


def main():
    t_start = time.time()
    samples, _ = build_dataset_cached(WEEKS)
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)
    train_val_samples = [s for s in samples if s.game_id in train_val_games]

    examples = build_examples(train_val_samples)
    roles = np.array([e.role for e in examples])
    tensors = build_tensors(examples)
    truth_np = tensors["truth"].numpy()
    mask_np = tensors["mask"].numpy()
    game_id_arr = tensors["game_id"]
    n_total = len(examples)

    sigma_a_raw = estimate_sigma_a(train_val_samples)
    sigma_a = calibrate_sigma_a(train_val_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    predictors = build_predictors_quadrature(sigma_a)

    print(f"computing {len(ALL_BASELINE_NAMES)} baselines out-of-fold "
         f"(= over the full train+val set, {n_total:,} tracks)...")
    base_arrays = {name: compute_base_arrays(train_val_samples, predictors[name], examples)
                  for name in ALL_BASELINE_NAMES}

    base_tks = base_arrays["tbl_kalman_speed"]
    base_vtb = base_arrays["velocity_turning_to_ball"]
    anchor_np = tensors["anchor"].numpy()
    base_anchor_repeat = np.repeat(anchor_np[:, None, :], truth_np.shape[1], axis=1)
    bases_t = {
        "K1": torch.from_numpy(base_tks),
        "K2": torch.from_numpy(base_vtb),
        "K3": torch.from_numpy(base_anchor_repeat),
    }
    bases_np = {"K1": base_tks, "K2": base_vtb, "K3": base_anchor_repeat}

    # ---------------------------------------------------------- baselines table
    print("\n" + "=" * 70)
    print("Full baseline table, out-of-fold, @1.9s and @3.0s -- overall and by role")
    print("=" * 70)
    baseline_rows = []
    for name in ALL_BASELINE_NAMES:
        arr = base_arrays[name]
        for step, label in ((PRIMARY_STEP, "1.9s"), (STEP_3S, "3.0s")):
            by_role = masked_error_by_role(arr, truth_np, mask_np, roles, step)
            row = {"predictor": name, "horizon": label}
            for k, (e, n) in by_role.items():
                row[f"{k}_err"] = round(e, 4)
                row[f"{k}_n"] = n
            baseline_rows.append(row)
    baseline_df = pd.DataFrame(baseline_rows)
    print(baseline_df.to_string(index=False))
    baseline_df.to_csv(OUTPUT_DIR / "tier1_taskl_baselines_by_role.csv", index=False)

    # ---------------------------------------------------------- K1/K2/K3, 5-fold CV
    n_folds = len(folds)
    per_fold_scores = {v: [] for v in ("K1", "K2", "K3")}
    per_fold_scores_3s = {v: [] for v in ("K1", "K2", "K3")}
    oof_pred = {v: np.full((n_total, truth_np.shape[1], 2), np.nan, dtype=np.float32)
               for v in ("K1", "K2", "K3")}
    oof_filled = np.zeros(n_total, dtype=bool)

    for i in range(n_folds):
        test_games = set(folds[i])
        val_games = set(folds[(i + 1) % n_folds])
        train_games = train_val_games - test_games - val_games
        test_idx = np.where(np.isin(game_id_arr, list(test_games)))[0]
        val_idx = np.where(np.isin(game_id_arr, list(val_games)))[0]
        train_idx = np.where(np.isin(game_id_arr, list(train_games)))[0]
        seed = SPLIT_SEED + i

        print(f"\n=== fold {i} ===")
        for variant in ("K1", "K2", "K3"):
            model, best_val, wall_clock, epochs_run = train_one(
                variant, bases_t[variant], tensors, train_idx, val_idx, seed, fold_id=i)
            model_out = predict_all(model, bases_t[variant], tensors, test_idx)
            pred = bases_np[variant][test_idx] + model_out
            oof_pred[variant][test_idx] = pred
            oof_filled[test_idx] = True

            err_p, n_p = masked_error_at_step(pred, truth_np[test_idx], mask_np[test_idx], PRIMARY_STEP)
            err_3, n_3 = masked_error_at_step(pred, truth_np[test_idx], mask_np[test_idx], STEP_3S)
            per_fold_scores[variant].append(err_p)
            per_fold_scores_3s[variant].append(err_3)
            print(f"  {variant}: epochs={epochs_run} wall_clock={wall_clock:.1f}s "
                 f"@1.9s={err_p:.4f} (n={n_p}) @3s={err_3:.4f} (n={n_3})")

    assert oof_filled.all()

    print("\n" + "=" * 70)
    print("K1/K2/K3 -- CV mean and fold spread")
    print("=" * 70)
    cv_rows = []
    for variant in ("K1", "K2", "K3"):
        s = np.array(per_fold_scores[variant])
        s3 = np.array(per_fold_scores_3s[variant])
        cv_rows.append({
            "predictor": variant, "cv_mean_1.9s": s.mean(), "cv_std_1.9s": s.std(ddof=1),
            "cv_min_1.9s": s.min(), "cv_max_1.9s": s.max(),
            "cv_mean_3s": s3.mean(), "cv_std_3s": s3.std(ddof=1),
        })
    cv_df = pd.DataFrame(cv_rows).round(4)
    print(cv_df.to_string(index=False))
    cv_df.to_csv(OUTPUT_DIR / "tier1_taskl_cv_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("K1/K2/K3 -- out-of-fold, by role (all folds combined), @1.9s")
    print("=" * 70)
    model_role_rows = []
    for variant in ("K1", "K2", "K3"):
        by_role = masked_error_by_role(oof_pred[variant], truth_np, mask_np, roles, PRIMARY_STEP)
        row = {"predictor": variant}
        for k, (e, n) in by_role.items():
            row[f"{k}_err"] = round(e, 4)
            row[f"{k}_n"] = n
        model_role_rows.append(row)
    model_role_df = pd.DataFrame(model_role_rows)
    print(model_role_df.to_string(index=False))
    model_role_df.to_csv(OUTPUT_DIR / "tier1_taskl_model_by_role.csv", index=False)

    print(f"\ntotal wall-clock: {time.time()-t_start:.1f}s")
    print("\nCV portion of Task L complete. Test split untouched. Stopping "
         "to confirm final-model/test methodology before proceeding.")


if __name__ == "__main__":
    main()
