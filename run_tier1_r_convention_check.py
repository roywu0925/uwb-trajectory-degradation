"""R-convention check (user-flagged modelling question, 2026-09-28).

    python run_tier1_r_convention_check.py

`run_tier1_sweep.set_filter_R`'s "floor" convention treats `sigma` (the
injected noise) as the FULL position measurement noise, with a 0.05
floor only to keep R invertible at sigma=0. That implicitly assumes the
input positions are physically exact before `sigma` is injected -- but
in `derived` mode, `sigma` is added on top of positions that already
carry NGS's own real localisation error (the UWB spec, ~0.15-0.3 yd,
already used elsewhere in this project as `config.KALMAN_R_POS`,
sigma=0.25). The "quadrature" convention accounts for that:

    floor:       sigma_pos = max(sigma, 0.05)
    quadrature:  sigma_pos = sqrt(KALMAN_R_POS + sigma**2)

At sigma=0.103 (Finding 1's crossing under "floor"), the two conventions
disagree sharply about what R the filter should use: 0.103 vs. sqrt(0.25^2
+ 0.103^2) = 0.270. A filter that understates R under-smooths and drifts
toward constant_velocity's own behaviour -- exactly the regime the
crossings live in.

This does NOT switch the convention. It reports both, side by side:
  - the clean-data (sigma=0) table at both horizons, both conventions
  - Finding 1's (kalman vs constant_velocity) and Finding 4's
    (tbl_kalman_speed vs constant_velocity) crossing sigmas, both
    conventions, primary horizon, reduced seed count (2, for tractability
    -- this is a comparative check, not a replacement for the 5-seed
    bootstrap already reported)

If the crossings move materially, that's a limitation to state, not
something to pick a winner on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import imm as imm_module
import kalman as kalman_module
from config import (CROSSOVER_SIGMAS, DEGRADE_BASE_SEED, DEGRADE_SIGMA_VEL_NGS,
                    DT, KALMAN_R_POS, OUTPUT_DIR)
from data_loader import build_samples
from degrade import corrupt_input
from evaluate import displacement_error, evaluate, summarise_at_step
from run_tier1_crossover import find_crossing
from run_tier1_sweep import build_predictors, frozen_sigma_a, load_clean

PRIMARY_STEP = 19
STEP_3S = 30
CHECK_SEEDS = 2
CONVENTIONS = ("floor", "quadrature")


def set_filter_R_convention(sigma: float, velocity_source: str, convention: str) -> None:
    if convention == "floor":
        sigma_pos = max(sigma, 0.05)
    elif convention == "quadrature":
        sigma_pos = np.sqrt(KALMAN_R_POS + sigma ** 2)
    else:
        raise ValueError(f"unknown convention {convention!r}")

    if velocity_source == "derived":
        sigma_vel = np.sqrt(2) * sigma_pos / DT
    elif velocity_source == "ngs":
        sigma_vel = DEGRADE_SIGMA_VEL_NGS
    else:
        raise ValueError(f"unknown velocity_source {velocity_source!r}")

    r_pos_vel = np.diag([sigma_pos**2, sigma_pos**2, sigma_vel**2, sigma_vel**2])
    kalman_module.R_POS = (sigma_pos ** 2) * np.eye(2)
    kalman_module.R_POS_VEL = r_pos_vel
    imm_module.R = r_pos_vel


def clean_table(sigma_a, clean_inp, clean_out, convention) -> pd.DataFrame:
    """sigma=0, both horizons -- seed-independent since no noise is
    injected, so a single evaluation suffices."""
    rng = np.random.default_rng(DEGRADE_BASE_SEED)
    degraded = corrupt_input(clean_inp, 0.0, rng, "derived")
    samples = build_samples(degraded, clean_out)
    set_filter_R_convention(0.0, "derived", convention)
    predictors = build_predictors(sigma_a)
    errors = evaluate(samples, predictors)

    rows = []
    for step, label in ((PRIMARY_STEP, "1.9s"), (STEP_3S, "3.0s")):
        t = summarise_at_step(errors, step)
        t.insert(0, "convention", convention)
        t.insert(1, "step_label", label)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def run_reduced_crossover(sigma_a, clean_inp, clean_out, convention,
                          seeds=range(CHECK_SEEDS)) -> pd.DataFrame:
    play_rows = []
    combos = [(s, seed) for s in CROSSOVER_SIGMAS for seed in seeds]
    for i, (sigma, seed) in enumerate(combos, 1):
        rng = np.random.default_rng(DEGRADE_BASE_SEED + seed)
        degraded = corrupt_input(clean_inp, sigma, rng, "derived")
        samples = build_samples(degraded, clean_out)
        set_filter_R_convention(sigma, "derived", convention)
        predictors = build_predictors(sigma_a)
        for sample in samples:
            if sample.horizon < PRIMARY_STEP:
                continue
            for name in ("constant_velocity", "kalman", "tbl_kalman_speed"):
                pred = predictors[name](sample, sample.horizon)
                err = displacement_error(pred, sample.truth)[:, PRIMARY_STEP - 1]
                play_rows.append((sigma, seed, sample.game_id, sample.play_id,
                                  name, float(np.mean(err))))
        print(f"  [{convention} {i}/{len(combos)}] sigma={sigma} seed={seed}")
    return pd.DataFrame(play_rows, columns=["sigma", "seed", "game_id", "play_id",
                                            "predictor", "mde"])


def crossing_from_play_level(play_level, predictor_a, predictor_b, sigma_grid):
    def matrix(name):
        per_play = (play_level[play_level["predictor"] == name]
                   .groupby(["game_id", "play_id", "sigma"])["mde"].mean().reset_index())
        plays = sorted(per_play[["game_id", "play_id"]]
                       .drop_duplicates().itertuples(index=False, name=None))
        pidx = {p: i for i, p in enumerate(plays)}
        sidx = {s: i for i, s in enumerate(sigma_grid)}
        mat = np.full((len(plays), len(sigma_grid)), np.nan)
        for row in per_play.itertuples(index=False):
            mat[pidx[(row.game_id, row.play_id)], sidx[row.sigma]] = row.mde
        return mat
    mat_a = matrix(predictor_a)
    mat_b = matrix(predictor_b)
    return find_crossing(sigma_grid, np.nanmean(mat_a, axis=0), np.nanmean(mat_b, axis=0))


def main():
    print("loading full 18-week dataset...")
    clean_inp, clean_out = load_clean()
    sigma_a = frozen_sigma_a(clean_inp, clean_out)

    print("\n" + "=" * 70)
    print("Clean-data (sigma=0) table, both conventions, both horizons")
    print("=" * 70)
    tables = [clean_table(sigma_a, clean_inp, clean_out, c) for c in CONVENTIONS]
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "tier1_r_convention_clean_table.csv", index=False)
    for step_label in ("1.9s", "3.0s"):
        print(f"\n--- {step_label} ---")
        sub = combined[combined["step_label"] == step_label]
        piv = sub.pivot(index="predictor", columns="convention", values="mean_error")
        piv["delta"] = piv["quadrature"] - piv["floor"]
        print(piv.sort_values("floor").round(3).to_string())

    print("\n" + "=" * 70)
    print(f"Crossing sigmas, primary horizon (1.9s), {CHECK_SEEDS} seeds, "
         f"both conventions")
    print("=" * 70)
    results = {}
    for convention in CONVENTIONS:
        play_level = run_reduced_crossover(sigma_a, clean_inp, clean_out, convention)
        play_level.to_csv(OUTPUT_DIR / f"tier1_r_convention_{convention}_playlevel.csv",
                          index=False)
        kalman_x = crossing_from_play_level(play_level, "kalman",
                                            "constant_velocity", CROSSOVER_SIGMAS)
        hybrid_x = crossing_from_play_level(play_level, "tbl_kalman_speed",
                                            "constant_velocity", CROSSOVER_SIGMAS)
        results[convention] = (kalman_x, hybrid_x)

    print("\nsummary:")
    print(f"{'convention':<14}{'kalman crossing':>20}{'tbl_kalman_speed crossing':>28}")
    for convention, (k, h) in results.items():
        k_s = f"{k:.4f}" if k is not None else "n/a"
        h_s = f"{h:.4f}" if h is not None else "n/a"
        print(f"{convention:<14}{k_s:>20}{h_s:>28}")

    if all(v[0] is not None for v in results.values()):
        k_floor, k_quad = results["floor"][0], results["quadrature"][0]
        print(f"\nkalman crossing moved by {k_quad - k_floor:+.4f} yd "
             f"({100*(k_quad-k_floor)/k_floor:+.1f}%)")
    if all(v[1] is not None for v in results.values()):
        h_floor, h_quad = results["floor"][1], results["quadrature"][1]
        print(f"tbl_kalman_speed crossing moved by {h_quad - h_floor:+.4f} yd "
             f"({100*(h_quad-h_floor)/h_floor:+.1f}%)")


if __name__ == "__main__":
    main()
