"""Answering a direct question: is sigma=0.934 (README's reported K5 vs.
tbl_kalman_speed test-split crossing) actually the point where K5 is first
beaten by the BEST of all 13 physics baselines, or only by a 3-baseline
subset?

run_tier1_test_degrade.py only ever scored constant_velocity,
tbl_kalman_speed and velocity_turning_to_ball under noise on the test
split. The other 10 baselines (constant_position, constant_acceleration,
damped_velocity, toward_ball_land, kalman, kalman_velocity, kalman_ca,
imm, tbl_kalman_pos, tbl_kalman_both) were never run under degradation on
test anywhere in this project. This script closes that gap: same test
games, same corruption seeds (DEGRADE_BASE_SEED_M, so the degraded inputs
are bit-identical to the existing tier1_test_crossover_playlevel.csv run),
same fine grid, same play-level methodology -- but for the 10 baselines
that were never measured.

Diagnostic/verification script, not part of the Reproducing pipeline.
Writes to its own output files; does not touch tier1_test_degrade_raw.csv
or tier1_test_crossover_playlevel.csv.

    PYTHONPATH=. python scripts/run_tier1_test_degrade_allbaselines.py  # run from the repo root
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from config import OUTPUT_DIR, WEEKS
from data_loader import add_kinematics, build_samples, load_raw, normalise
from degrade import corrupt_input
from kalman import calibrate_sigma_a, estimate_sigma_a
from model import build_examples
from run_tier1_taskk import build_tensors, masked_error_at_step
from run_tier1_taskp import build_predictors_quadrature
from run_tier1_r_convention_check import set_filter_R_convention

PRIMARY_STEP = 19
VELOCITY_SOURCE = "derived"
DEGRADE_BASE_SEED_M = 20260929
FINE_GRID = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
SEEDS = [0, 1, 2, 3, 4]

NEW_PREDICTOR_NAMES = [
    "constant_position", "constant_acceleration", "damped_velocity",
    "toward_ball_land",
    "kalman", "kalman_velocity", "kalman_ca", "imm",
    "tbl_kalman_pos", "tbl_kalman_both",
]


def main():
    with open(OUTPUT_DIR / "splits.json") as f:
        splits = json.load(f)
    test_games = set(splits["test"])
    train_val_games = set(splits["train"] + splits["val"])

    print("loading raw input/output, test games only...")
    inp, out = load_raw(weeks=WEEKS)
    inp = inp[inp["game_id"].isin(test_games)].reset_index(drop=True)
    out = out[out["game_id"].isin(test_games)].reset_index(drop=True)
    inp, out = normalise(inp, out)
    clean_test_inp = add_kinematics(inp)
    clean_test_out = out

    print("loading raw input/output, train+val games only, for frozen sigma_a...")
    inp_tv, out_tv = load_raw(weeks=WEEKS)
    inp_tv = inp_tv[inp_tv["game_id"].isin(train_val_games)].reset_index(drop=True)
    out_tv = out_tv[out_tv["game_id"].isin(train_val_games)].reset_index(drop=True)
    inp_tv, out_tv = normalise(inp_tv, out_tv)
    inp_tv = add_kinematics(inp_tv)
    tv_samples_for_sigma = build_samples(inp_tv, out_tv)
    sigma_a_raw = estimate_sigma_a(tv_samples_for_sigma)
    sigma_a = calibrate_sigma_a(tv_samples_for_sigma, sigma_a_hi=sigma_a_raw, target=2.0)
    print(f"frozen sigma_a = {sigma_a:.4f}")

    play_rows = []
    summary_rows = []
    combos = [(s, 0) if s == 0.0 else (s, seed) for s in [0.0] + FINE_GRID
             for seed in (SEEDS if s > 0 else [0])]
    col = PRIMARY_STEP - 1

    for k, (sigma, seed) in enumerate(combos, 1):
        print(f"[{k}/{len(combos)}] sigma={sigma} seed={seed}")
        rng = np.random.default_rng(DEGRADE_BASE_SEED_M + seed)
        degraded_inp = corrupt_input(clean_test_inp, sigma, rng, VELOCITY_SOURCE)
        degraded_samples = build_samples(degraded_inp, clean_test_out)

        examples_std = build_examples(degraded_samples)
        tensors_std = build_tensors(examples_std)
        truth_std = tensors_std["truth"].numpy()
        mask_std = tensors_std["mask"].numpy()
        anchor_std = tensors_std["anchor"].numpy()
        base_anchor_std = np.repeat(anchor_std[:, None, :], truth_std.shape[1], axis=1)
        meta_std = [(e.game_id, e.play_id) for e in examples_std]

        set_filter_R_convention(sigma, VELOCITY_SOURCE, "quadrature")
        predictors = build_predictors_quadrature(sigma_a)

        for name in NEW_PREDICTOR_NAMES:
            fn = predictors[name]
            arr = np.zeros_like(base_anchor_std)
            pos = 0
            for sample in degraded_samples:
                n = len(sample.state)
                arr[pos:pos + n] = fn(sample, truth_std.shape[1])
                pos += n
            err, n_v = masked_error_at_step(arr, truth_std, mask_std, PRIMARY_STEP)
            summary_rows.append({"sigma": sigma, "seed": seed, "predictor": name,
                                 "error": err, "n": n_v})

            valid = mask_std[:, col] > 0
            for local_j in np.where(valid)[0]:
                d = float(np.linalg.norm(arr[local_j, col] - truth_std[local_j, col]))
                gid, pid = meta_std[local_j]
                play_rows.append((sigma, seed, PRIMARY_STEP, gid, pid, name, d))

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "tier1_test_degrade_allbaselines_raw.csv", index=False)
    pivot = df_summary.groupby(["sigma", "predictor"])["error"].mean().reset_index()
    pivot = pivot.pivot(index="sigma", columns="predictor", values="error").round(4)
    print("\n" + "=" * 70)
    print("TEST-SPLIT degradation, 10 previously-unmeasured baselines, mean error @1.9s")
    print("=" * 70)
    print(pivot.to_string())
    pivot.to_csv(OUTPUT_DIR / "tier1_test_degrade_allbaselines_summary.csv")

    play_level = pd.DataFrame(play_rows, columns=[
        "sigma", "seed", "step", "game_id", "play_id", "predictor", "mde"])
    play_level.to_csv(OUTPUT_DIR / "tier1_test_degrade_allbaselines_playlevel.csv", index=False)
    print(f"\nwrote {len(play_level):,} play-level rows")


if __name__ == "__main__":
    main()
