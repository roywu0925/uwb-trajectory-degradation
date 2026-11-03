"""Tier 1 Task M (TIER1_SPEC_6.md): sensor-degradation robustness of the
three trained model variants, CV folds only -- test split never touched.

Uses the CLEAN-trained checkpoints from Task L (outputs/checkpoints/
{K1,K2,K3}_fold{0..4}.pt) unchanged -- no retraining. For each
sigma_player level, corrupts train+val positions only (same mechanism as
Task B/C), rebuilds features from the corrupted data, and runs the
frozen clean-trained model forward. Baselines are rebuilt at each sigma
under the `quadrature` R convention (the project's standing convention
since TIER1_SPEC_5.md), using the same frozen sigma_a as everywhere else
in Tier 1.

Falsifiable expectation stated in TIER1_SPEC_6.md: K3 (no physics base)
degrades fastest, K2 (raw-speed base) next, K1 (Kalman-smoothed base)
slowest.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from config import DT, OUTPUT_DIR, PLAYER_SIGMAS, SPLIT_SEED, WEEKS
from data_loader import add_kinematics, build_samples, normalise
from degrade import corrupt_input
from model import GRUResidualModel, build_examples
from run_tier1_taskk import build_tensors, compute_base_arrays, masked_error_at_step, predict_all
from run_tier1_taskp import build_predictors_quadrature
from run_tier1_r_convention_check import set_filter_R_convention

PRIMARY_STEP = 19
VELOCITY_SOURCE = "derived"
GRID_SEEDS = 3
DEGRADE_BASE_SEED_M = 20260929   # independent stream, this check only

BASELINE_SUBSET = ["constant_velocity", "velocity_turning_to_ball", "kalman", "tbl_kalman_speed"]


def load_train_val_raw(train_val_games: set):
    """Same load->normalise->add_kinematics steps as run_tier1_sweep.
    load_clean (usecols-trimmed, RENAME applied), restricted to
    train+val games right after loading -- the test split's raw rows
    never reach normalise/add_kinematics or any later step."""
    from data_loader import load_raw
    print("loading raw input/output, all weeks...")
    inp, out = load_raw(weeks=WEEKS)
    inp = inp[inp["game_id"].isin(train_val_games)].reset_index(drop=True)
    out = out[out["game_id"].isin(train_val_games)].reset_index(drop=True)
    inp, out = normalise(inp, out)
    inp = add_kinematics(inp)
    print(f"  {len(inp):,} input rows, {len(out):,} output rows (train+val games only)")
    return inp, out


def evaluate_models_at_sigma(sigma, seed, folds, tensors, examples, base_arrays_model,
                             game_id_arr, checkpoints):
    """Run the frozen clean-trained K1/K2/K3 checkpoints on this sigma's
    degraded tensors, out-of-fold (each fold's own checkpoint scores its
    own held-out fold)."""
    truth_np = tensors["truth"].numpy()
    mask_np = tensors["mask"].numpy()
    n = len(examples)
    oof_pred = {v: np.full((n, truth_np.shape[1], 2), np.nan, dtype=np.float32)
               for v in ("K1", "K2", "K3")}
    for i, fold_games in enumerate(folds):
        test_idx = np.where(np.isin(game_id_arr, fold_games))[0]
        for variant in ("K1", "K2", "K3"):
            model = checkpoints[(variant, i)]
            base_t = torch.from_numpy(base_arrays_model[variant])
            model_out = predict_all(model, base_t, tensors, test_idx)
            oof_pred[variant][test_idx] = base_arrays_model[variant][test_idx] + model_out

    rows = []
    for variant in ("K1", "K2", "K3"):
        err, n_v = masked_error_at_step(oof_pred[variant], truth_np, mask_np, PRIMARY_STEP)
        rows.append({"sigma": sigma, "seed": seed, "predictor": variant, "error": err, "n": n_v})
    return rows


def main():
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)

    clean_inp, clean_out = load_train_val_raw(train_val_games)

    # frozen sigma_a: reuse Task P's value directly, do not re-estimate here
    clean_samples_for_sigma = build_samples(clean_inp, clean_out)
    from kalman import calibrate_sigma_a, estimate_sigma_a
    sigma_a_raw = estimate_sigma_a(clean_samples_for_sigma)
    sigma_a = calibrate_sigma_a(clean_samples_for_sigma, sigma_a_hi=sigma_a_raw, target=2.0)
    print(f"frozen sigma_a = {sigma_a:.4f} (train+val, clean, matches Task P/K/L)")

    # load all 15 checkpoints once
    print("loading Task L checkpoints...")
    checkpoints = {}
    for variant in ("K1", "K2", "K3"):
        for i in range(len(folds)):
            ckpt = torch.load(OUTPUT_DIR / "checkpoints" / f"{variant}_fold{i}.pt", weights_only=False)
            model = GRUResidualModel(hidden=ckpt["config"]["hidden"])
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            checkpoints[(variant, i)] = model

    combos = []
    for sigma in PLAYER_SIGMAS:
        if sigma == 0.0:
            combos.append((sigma, 0))
        else:
            for seed in range(GRID_SEEDS):
                combos.append((sigma, seed))

    all_rows = []
    for k, (sigma, seed) in enumerate(combos, 1):
        print(f"\n[{k}/{len(combos)}] sigma={sigma} seed={seed}")
        rng = np.random.default_rng(DEGRADE_BASE_SEED_M + seed)
        degraded_inp = corrupt_input(clean_inp, sigma, rng, VELOCITY_SOURCE)
        degraded_samples = build_samples(degraded_inp, clean_out)

        examples = build_examples(degraded_samples)
        tensors = build_tensors(examples)
        game_id_arr = tensors["game_id"]

        set_filter_R_convention(sigma, VELOCITY_SOURCE, "quadrature")
        predictors = build_predictors_quadrature(sigma_a)
        base_arrays_model = {
            "K1": compute_base_arrays(degraded_samples, predictors["tbl_kalman_speed"], examples),
            "K2": compute_base_arrays(degraded_samples, predictors["velocity_turning_to_ball"], examples),
            "K3": np.repeat(tensors["anchor"].numpy()[:, None, :], tensors["truth"].shape[1], axis=1),
        }

        model_rows = evaluate_models_at_sigma(
            sigma, seed, folds, tensors, examples, base_arrays_model, game_id_arr, checkpoints)
        all_rows.extend(model_rows)

        truth_np = tensors["truth"].numpy()
        mask_np = tensors["mask"].numpy()
        for name in BASELINE_SUBSET:
            arr = compute_base_arrays(degraded_samples, predictors[name], examples)
            err, n_v = masked_error_at_step(arr, truth_np, mask_np, PRIMARY_STEP)
            all_rows.append({"sigma": sigma, "seed": seed, "predictor": name, "error": err, "n": n_v})

        for row in model_rows:
            print(f"  {row['predictor']}: {row['error']:.4f} (n={row['n']})")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "tier1_taskm_raw.csv", index=False)

    summary = (df.groupby(["sigma", "predictor"])["error"]
              .agg(mean="mean", std="std", n_seeds="size").reset_index())
    pivot = summary.pivot(index="sigma", columns="predictor", values="mean").round(4)
    print("\n" + "=" * 70)
    print("Task M -- mean error @1.9s vs sigma_player (averaged over seeds)")
    print("=" * 70)
    print(pivot.to_string())
    pivot.to_csv(OUTPUT_DIR / "tier1_taskm_summary.csv")

    print("\nrelative degradation (error / clean error), K1/K2/K3:")
    rel = pivot[["K1", "K2", "K3"]] / pivot[["K1", "K2", "K3"]].iloc[0]
    print(rel.round(3).to_string())


if __name__ == "__main__":
    main()
