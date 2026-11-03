"""Tier 1 Task O: restate every finding at the honest primary horizon,
with 3s as a secondary metric, on the full 18-week data (TIER1_SPEC_5.md).

    PYTHONPATH=. python scripts/run_tier1_taskO.py  # run from the repo root

Runs the crossover grid and the coarse grid (derived mode only, same
grids/seeds as Parts 2-3, no tuning) on all 2,729 plays, capturing
play-level data at BOTH the new primary horizon (step 19, 1.9s, chosen
by run_tier1_taskn.py at >=50% track coverage) and the old 3s secondary
(step 30) -- Finding 1's crossing-sigma bootstrap needs play-level data
at whichever horizon it's being asked about, and TIER1_SPEC_5.md asks
for both.

Writes outputs/tier1_taskO_crossover.csv, outputs/tier1_taskO_coarse.csv,
and prints every number needed to restate Findings 1-4 with
(n_plays, n_tracks, coverage) attached, at both horizons.
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

PRIMARY_STEP = 19   # 1.9s, from run_tier1_taskn.py's coverage audit
SECONDARY_STEP = 30  # 3.0s, the old headline horizon


def coverage_table(tidy: pd.DataFrame, step: int) -> pd.DataFrame:
    """(predictor, n, coverage) at a fixed sigma=0.0, one fixed step --
    coverage is identical across predictors (same underlying samples),
    included per-row anyway so no table is ever silent about it."""
    at_zero = tidy[(tidy["sigma"] == 0.0) & (tidy["step"] == step)]
    n_by_pred = at_zero.groupby("predictor")["n"].sum()
    cov_by_pred = at_zero.groupby("predictor")["coverage"].mean()
    return pd.DataFrame({"n": n_by_pred, "coverage": cov_by_pred})


def mde_table(tidy: pd.DataFrame, step: int, predictors: list) -> pd.DataFrame:
    at_step = tidy[tidy["step"] == step]
    per_run = (at_step.groupby(["sigma", "seed", "predictor"])
              .apply(lambda g: np.average(g["mde"], weights=g["n"]),
                     include_groups=False)
              .rename("mde").reset_index())
    summary = per_run.groupby(["sigma", "predictor"])["mde"].mean().unstack("predictor")
    return summary[[p for p in predictors if p in summary.columns]]


def main():
    t_start = time.time()
    print("loading full 18-week dataset...")
    clean_inp, clean_out = load_clean()
    sigma_a = frozen_sigma_a(clean_inp, clean_out)

    print(f"\nrunning crossover grid: {len(CROSSOVER_SIGMAS)} sigmas x "
         f"{CROSSOVER_SEEDS} seeds, derived mode only, capturing play-level "
         f"data at steps {PRIMARY_STEP} (1.9s) and {SECONDARY_STEP} (3.0s)")
    tidy_x, play_level, signed_error = run_crossover_sweep(
        sigma_a, clean_inp, clean_out,
        play_level_steps=(PRIMARY_STEP, SECONDARY_STEP))
    tidy_x.to_csv(OUTPUT_DIR / "tier1_taskO_crossover.csv", index=False)
    print(f"wrote {OUTPUT_DIR / 'tier1_taskO_crossover.csv'} ({len(tidy_x):,} rows)")

    print(f"\nrunning coarse grid: {len(NOISE_SIGMAS)} sigmas x {N_SEEDS} "
         f"seeds, derived mode only, full 18-week data")
    tidy_c, signed_error_c = run_sweep(NOISE_SIGMAS, range(N_SEEDS),
                                       ["derived"], sigma_a, clean_inp, clean_out)
    tidy_c.to_csv(OUTPUT_DIR / "tier1_taskO_coarse.csv", index=False)
    print(f"wrote {OUTPUT_DIR / 'tier1_taskO_coarse.csv'}")

    print(f"\ntotal sweep time: {time.time() - t_start:.0f}s")

    # ---- Finding 1: crossing sigma, at both horizons
    print("\n" + "=" * 70)
    print("Finding 1 -- crossover sigma, both horizons, full 18-week data")
    print("=" * 70)
    for label, step in (("primary (1.9s)", PRIMARY_STEP), ("secondary (3.0s)", SECONDARY_STEP)):
        cov = coverage_table(tidy_x, step)
        n_cv = int(cov.loc["constant_velocity", "n"]) if "constant_velocity" in cov.index else None
        point = point_estimate_crossing(play_level, CROSSOVER_SIGMAS, step)
        boot = bootstrap_crossing(play_level, CROSSOVER_SIGMAS, step, n_resamples=1000)
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (float("nan"), float("nan")))
        print(f"\n  {label}: n={n_cv} tracks/sigma, point={point}, "
             f"95% CI [{lo:.4f}, {hi:.4f}] (width {hi - lo:.4f}), "
             f"{len(boot)}/1000 resamples found a crossing")

    # ---- Findings 2 & 4: crossover-grid tables, both horizons
    cols = ["toward_ball_land", "tbl_kalman_speed", "tbl_kalman_pos",
           "tbl_kalman_both", "kalman", "kalman_velocity",
           "constant_velocity", "velocity_turning_to_ball", "imm"]
    for label, step in (("primary (1.9s)", PRIMARY_STEP), ("secondary (3.0s)", SECONDARY_STEP)):
        print("\n" + "=" * 70)
        print(f"Findings 2 & 4 -- crossover-grid MDE, {label}")
        cov = coverage_table(tidy_x, step)
        print(f"coverage at this step (sigma=0): n={int(cov['n'].iloc[0])}, "
             f"{cov['coverage'].iloc[0]:.1%}")
        print("=" * 70)
        print(mde_table(tidy_x, step, cols).round(3).to_string())

    # ---- Findings 3 & 4: coarse-grid tables, both horizons
    cols_c = ["toward_ball_land", "tbl_kalman_speed", "tbl_kalman_both",
             "kalman", "kalman_velocity", "constant_velocity", "imm",
             "kalman_ca"]
    for label, step in (("primary (1.9s)", PRIMARY_STEP), ("secondary (3.0s)", SECONDARY_STEP)):
        print("\n" + "=" * 70)
        print(f"Findings 3 & 4 -- coarse-grid MDE (0-5 yd), {label}")
        cov = coverage_table(tidy_c, step)
        print(f"coverage at this step (sigma=0): n={int(cov['n'].iloc[0])}, "
             f"{cov['coverage'].iloc[0]:.1%}")
        print("=" * 70)
        print(mde_table(tidy_c, step, cols_c).round(2).to_string())


if __name__ == "__main__":
    main()
