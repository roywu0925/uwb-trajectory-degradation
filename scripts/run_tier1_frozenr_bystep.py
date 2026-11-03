"""Item 1 follow-up: error-vs-step for `kalman`, adaptive vs frozen R, at
sigma levels where the aggregate @1.9s ratio showed frozen R winning
(sigma=1.0, 5.0). Tests the "wrong dynamics model + correctly-large R
runs open-loop and lags on turns" hypothesis: if it holds, adaptive R
should win at short steps (near-constant-velocity is nearly right close
to the anchor) and lose at long steps (lag compounds). No crossover
would mean something else is going on.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import DT, OUTPUT_DIR, WEEKS
from data_loader import build_samples
from degrade import corrupt_input
from kalman import calibrate_sigma_a, estimate_sigma_a
from model import build_examples
from run_tier1_taskk import build_tensors, compute_base_arrays
from run_tier1_taskp import build_predictors_quadrature
from run_tier1_r_convention_check import set_filter_R_convention
from run_tier1_taskm import load_train_val_raw

VELOCITY_SOURCE = "derived"
DEGRADE_BASE_SEED_M = 20260929
SIGMAS_TO_CHECK = [1.0, 5.0]
SEEDS = [0, 1, 2]
MAX_STEP = 30


def error_by_step(pred, truth, mask, max_step):
    rows = []
    for step in range(1, max_step + 1):
        col = step - 1
        valid = mask[:, col] > 0
        if valid.sum() == 0:
            continue
        err = np.linalg.norm(pred[valid, col] - truth[valid, col], axis=-1).mean()
        rows.append((step, float(err), int(valid.sum())))
    return rows


def main():
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)

    clean_inp, clean_out = load_train_val_raw(train_val_games)
    clean_samples_for_sigma = build_samples(clean_inp, clean_out)
    sigma_a_raw = estimate_sigma_a(clean_samples_for_sigma)
    sigma_a = calibrate_sigma_a(clean_samples_for_sigma, sigma_a_hi=sigma_a_raw, target=2.0)

    all_rows = []
    for sigma in SIGMAS_TO_CHECK:
        for seed in SEEDS:
            rng = np.random.default_rng(DEGRADE_BASE_SEED_M + seed)
            degraded_inp = corrupt_input(clean_inp, sigma, rng, VELOCITY_SOURCE)
            degraded_samples = build_samples(degraded_inp, clean_out)
            examples = build_examples(degraded_samples)
            tensors = build_tensors(examples)
            truth_np = tensors["truth"].numpy()
            mask_np = tensors["mask"].numpy()

            predictors = build_predictors_quadrature(sigma_a)

            set_filter_R_convention(sigma, VELOCITY_SOURCE, "quadrature")
            arr_adaptive = compute_base_arrays(degraded_samples, predictors["kalman"], examples)
            for step, err, n in error_by_step(arr_adaptive, truth_np, mask_np, MAX_STEP):
                all_rows.append({"sigma": sigma, "seed": seed, "r_mode": "adaptive",
                                "step": step, "error": err, "n": n})

            set_filter_R_convention(0.0, VELOCITY_SOURCE, "quadrature")
            arr_frozen = compute_base_arrays(degraded_samples, predictors["kalman"], examples)
            for step, err, n in error_by_step(arr_frozen, truth_np, mask_np, MAX_STEP):
                all_rows.append({"sigma": sigma, "seed": seed, "r_mode": "frozen",
                                "step": step, "error": err, "n": n})
            print(f"sigma={sigma} seed={seed} done")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "tier1_frozenr_bystep_raw.csv", index=False)
    summary = df.groupby(["sigma", "r_mode", "step"])["error"].mean().reset_index()
    pivot = summary.pivot_table(index=["sigma", "step"], columns="r_mode", values="error").round(4)
    pivot["ratio"] = (pivot["frozen"] / pivot["adaptive"]).round(3)
    print(pivot.to_string())
    pivot.to_csv(OUTPUT_DIR / "tier1_frozenr_bystep_summary.csv")


if __name__ == "__main__":
    main()
