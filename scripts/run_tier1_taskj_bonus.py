"""Tier 1 Task J 'free bonus': re-run the existing Tier 0/1 findings on
the full 18-week data (TIER1_SPEC_4.md).

    PYTHONPATH=. python scripts/run_tier1_taskj_bonus.py  # run from the repo root

Reuses the exact sweep machinery from run_tier1_sweep.py /
run_tier1_crossover.py, unchanged, just pointed at config.WEEKS now
covering all 18 weeks instead of one. Restricted to derived mode only
(the mode every one of the four findings' headline numbers actually
uses) to keep this bonus check's runtime bounded -- it does not touch
`ngs` mode, Q, R, tau, or either grid's values.

Writes outputs/tier1_taskj_bonus_crossover.csv,
outputs/tier1_taskj_bonus_coarse.csv, and prints the four findings'
headline numbers side by side with their Part 1-3 (week-1-only) values.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from config import (CROSSOVER_SEEDS, CROSSOVER_SIGMAS, DT, N_SEEDS,
                    NOISE_SIGMAS, OUTPUT_DIR)
from run_tier1_crossover import (bootstrap_crossing, point_estimate_crossing,
                                 run_crossover_sweep)
from run_tier1_sweep import frozen_sigma_a, load_clean, run_sweep

STEP_3S = round(3.0 / DT)


def main():
    t_start = time.time()
    print("loading full 18-week dataset...")
    clean_inp, clean_out = load_clean()
    sigma_a = frozen_sigma_a(clean_inp, clean_out)

    print(f"\nrunning crossover grid: {len(CROSSOVER_SIGMAS)} sigmas x "
         f"{CROSSOVER_SEEDS} seeds, derived mode only, full 18-week data")
    tidy_x, play_level, signed_error = run_crossover_sweep(
        sigma_a, clean_inp, clean_out)
    tidy_x.to_csv(OUTPUT_DIR / "tier1_taskj_bonus_crossover.csv", index=False)
    print(f"wrote {OUTPUT_DIR / 'tier1_taskj_bonus_crossover.csv'}")

    point = point_estimate_crossing(play_level, CROSSOVER_SIGMAS)
    print(f"\npoint-estimate crossing sigma: {point}")
    boot = bootstrap_crossing(play_level, CROSSOVER_SIGMAS, n_resamples=1000)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"bootstrap: {len(boot)}/1000 resamples found a crossing")
    print(f"95% interval: [{lo:.4f}, {hi:.4f}] yd  (width {hi - lo:.4f})")

    print(f"\nrunning coarse grid: {len(NOISE_SIGMAS)} sigmas x {N_SEEDS} "
         f"seeds, derived mode only, full 18-week data")
    tidy_c, signed_error_c = run_sweep(NOISE_SIGMAS, range(N_SEEDS),
                                       ["derived"], sigma_a, clean_inp, clean_out)
    tidy_c.to_csv(OUTPUT_DIR / "tier1_taskj_bonus_coarse.csv", index=False)
    print(f"wrote {OUTPUT_DIR / 'tier1_taskj_bonus_coarse.csv'}")

    print(f"\ntotal bonus re-run time: {time.time() - t_start:.0f}s")

    # ---- Finding 1: crossing sigma
    print("\n" + "=" * 70)
    print("Finding 1 -- crossover: week-1 (128 plays) vs full (18 weeks)")
    print("=" * 70)
    print("week-1:  point=0.0292 yd, 95% CI [0.0041, 0.0680] (width 0.0638)")
    print(f"full:    point={point:.4f} yd, 95% CI [{lo:.4f}, {hi:.4f}] "
         f"(width {hi - lo:.4f})")

    # ---- Finding 2 / 4: crossover-grid table
    print("\n" + "=" * 70)
    print("Findings 2 & 4 -- crossover-grid 3s MDE, full 18-week data")
    print("=" * 70)
    at3s = tidy_x[tidy_x["step"] == STEP_3S]
    per_run = (at3s.groupby(["sigma", "seed", "predictor"])
              .apply(lambda g: np.average(g["mde"], weights=g["n"]),
                     include_groups=False)
              .rename("mde").reset_index())
    summary = per_run.groupby(["sigma", "predictor"])["mde"].mean().unstack("predictor")
    cols = ["toward_ball_land", "tbl_kalman_speed", "tbl_kalman_pos",
           "tbl_kalman_both", "kalman", "kalman_velocity",
           "constant_velocity", "velocity_turning_to_ball"]
    print(summary[cols].round(3).to_string())

    # ---- Finding 3 / 4: coarse-grid table (full 0-5 range)
    print("\n" + "=" * 70)
    print("Findings 3 & 4 -- coarse-grid 3s MDE, full 18-week data")
    print("=" * 70)
    at3s_c = tidy_c[tidy_c["step"] == STEP_3S]
    per_run_c = (at3s_c.groupby(["sigma", "seed", "predictor"])
                .apply(lambda g: np.average(g["mde"], weights=g["n"]),
                       include_groups=False)
                .rename("mde").reset_index())
    summary_c = per_run_c.groupby(["sigma", "predictor"])["mde"].mean().unstack("predictor")
    cols_c = ["toward_ball_land", "tbl_kalman_speed", "tbl_kalman_both",
             "kalman", "kalman_velocity", "constant_velocity", "imm",
             "kalman_ca"]
    print(summary_c[cols_c].round(2).to_string())


if __name__ == "__main__":
    main()
