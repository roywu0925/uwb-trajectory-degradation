"""Frozen-R fairness check (TIER1_SPEC_6.md follow-up): in the degradation
sweep, the Kalman family's R is updated to the CURRENT sigma at every
grid point (set_filter_R_convention(sigma, ...)) -- the filter is TOLD
the noise level. The learned models get no such update. This re-runs
the same sigma_player grid with R FROZEN at the clean value
(sigma_NGS=0.25, i.e. whatever set_filter_R_convention(0.0, ...) sets),
never updated per sigma, for the 7 Kalman-family predictors, alongside
the already-recorded adaptive-R numbers. constant_velocity and
velocity_turning_to_ball use no R at all and are included unchanged, as
a control that the harness did nothing else.

Adaptive R stays the headline/recommended convention regardless of this
result -- a real deployment knows its sensor's spec (it's on the
datasheet). This is a diagnostic that isolates the mechanism, not a
proposed replacement configuration.

Does not touch Q, the grid, or anything about the models. CV folds
only (same train+val games as every other Task M script); test
untouched.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from baselines import BASELINES
from config import OUTPUT_DIR, PLAYER_SIGMAS, WEEKS
from data_loader import build_samples
from degrade import corrupt_input
from kalman import build_Q, calibrate_sigma_a, estimate_sigma_a
from model import build_examples
from run_tier1_taskk import compute_base_arrays, masked_error_at_step, build_tensors
from run_tier1_taskp import build_predictors_quadrature
from run_tier1_r_convention_check import set_filter_R_convention
from run_tier1_taskm import load_train_val_raw

PRIMARY_STEP = 19
VELOCITY_SOURCE = "derived"
GRID_SEEDS = 3
DEGRADE_BASE_SEED_M = 20260929

KALMAN_FAMILY = ["kalman", "kalman_velocity", "kalman_ca", "imm",
                "tbl_kalman_pos", "tbl_kalman_speed", "tbl_kalman_both"]
CONTROLS = ["constant_velocity", "velocity_turning_to_ball"]


def main():
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)

    clean_inp, clean_out = load_train_val_raw(train_val_games)
    clean_samples_for_sigma = build_samples(clean_inp, clean_out)
    sigma_a_raw = estimate_sigma_a(clean_samples_for_sigma)
    sigma_a = calibrate_sigma_a(clean_samples_for_sigma, sigma_a_hi=sigma_a_raw, target=2.0)
    print(f"frozen sigma_a = {sigma_a:.4f}")

    combos = []
    for sigma in PLAYER_SIGMAS:
        if sigma == 0.0:
            combos.append((sigma, 0))
        else:
            for seed in range(GRID_SEEDS):
                combos.append((sigma, seed))

    all_rows = []
    for k, (sigma, seed) in enumerate(combos, 1):
        print(f"[{k}/{len(combos)}] sigma={sigma} seed={seed}")
        rng = np.random.default_rng(DEGRADE_BASE_SEED_M + seed)
        degraded_inp = corrupt_input(clean_inp, sigma, rng, VELOCITY_SOURCE)
        degraded_samples = build_samples(degraded_inp, clean_out)
        examples = build_examples(degraded_samples)
        tensors = build_tensors(examples)
        truth_np = tensors["truth"].numpy()
        mask_np = tensors["mask"].numpy()

        predictors = build_predictors_quadrature(sigma_a)

        # -------- adaptive R: told the current sigma
        set_filter_R_convention(sigma, VELOCITY_SOURCE, "quadrature")
        for name in KALMAN_FAMILY:
            arr = compute_base_arrays(degraded_samples, predictors[name], examples)
            err, n_v = masked_error_at_step(arr, truth_np, mask_np, PRIMARY_STEP)
            all_rows.append({"sigma": sigma, "seed": seed, "predictor": name,
                            "r_mode": "adaptive", "error": err, "n": n_v})

        # -------- frozen R: always the clean (sigma=0) value, never updated
        set_filter_R_convention(0.0, VELOCITY_SOURCE, "quadrature")
        for name in KALMAN_FAMILY:
            arr = compute_base_arrays(degraded_samples, predictors[name], examples)
            err, n_v = masked_error_at_step(arr, truth_np, mask_np, PRIMARY_STEP)
            all_rows.append({"sigma": sigma, "seed": seed, "predictor": name,
                            "r_mode": "frozen", "error": err, "n": n_v})

        # -------- controls: no R at all, should be identical in both "modes"
        for name in CONTROLS:
            arr = compute_base_arrays(degraded_samples, BASELINES[name], examples)
            err, n_v = masked_error_at_step(arr, truth_np, mask_np, PRIMARY_STEP)
            all_rows.append({"sigma": sigma, "seed": seed, "predictor": name,
                            "r_mode": "n/a", "error": err, "n": n_v})

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "tier1_frozenr_raw.csv", index=False)

    summary = df.groupby(["sigma", "predictor", "r_mode"])["error"].mean().reset_index()
    pivot = summary.pivot_table(index=["predictor", "sigma"], columns="r_mode", values="error")
    if "adaptive" in pivot.columns and "frozen" in pivot.columns:
        pivot["ratio (frozen/adaptive)"] = (pivot["frozen"] / pivot["adaptive"]).round(3)
    pivot = pivot.round(4)
    print("\n" + "=" * 70)
    print("Frozen-R vs adaptive-R, @1.9s, mean over seeds")
    print("=" * 70)
    print(pivot.to_string())
    pivot.to_csv(OUTPUT_DIR / "tier1_frozenr_summary.csv")


if __name__ == "__main__":
    main()
