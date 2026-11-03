"""Tier 1 Task D/E/F: the crossover sweep (TIER1_SPEC_2.md).

    python run_tier1_crossover.py

Task B's coarse grid ([0, 0.2, 0.5, 1, 2, 5]) located the
constant_velocity/kalman crossing only as "somewhere below 0.2" -- below
NGS's own published accuracy range. This runs a fine grid
(config.CROSSOVER_SIGMAS) over that region, derived mode only, all ten
predictors, and:

  - locates the crossing sigma by linear interpolation between the
    bracketing grid points
  - puts a 95% interval on it via a play-level bootstrap (1000
    resamples) -- tracks within a play share a throw and a landing
    point, so the resampling unit is the play, not the player-track
  - checks whether toward_ball_land is the most noise-robust method, as
    predicted (its direction comes from geometry, not differencing)
  - cross-checks the measured constant_velocity curve against the
    closed-form noise-propagation prediction (Task E)
  - writes the two Task F figures

Writes outputs/tier1_crossover.csv, outputs/tier1_crossover.png,
outputs/tier1_robustness_full.png.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (CROSSOVER_SEEDS, CROSSOVER_SIGMAS, DEGRADE_BASE_SEED, DT,
                    NOISE_SIGMAS, OUTPUT_DIR)
from data_loader import build_samples
from degrade import corrupt_input
from evaluate import displacement_error, evaluate
from run_tier1_sweep import (SIGNED_ERROR_PREDICTORS, build_predictors,
                             compute_signed_error, frozen_sigma_a, load_clean,
                             set_filter_R, summarize_run)

STEP_3S = round(3.0 / DT)

MAIN_FIGURE_PREDICTORS = ["constant_velocity", "kalman", "kalman_velocity",
                          "imm", "toward_ball_land", "velocity_turning_to_ball",
                          "tbl_kalman_speed", "tbl_kalman_pos", "tbl_kalman_both"]
NGS_SPEC_RANGE = (0.15, 0.30)


# ------------------------------------------------------------------ sweep

def run_crossover_sweep(sigma_a: float, clean_inp: pd.DataFrame,
                        clean_out: pd.DataFrame,
                        play_level_steps: tuple = (STEP_3S,)
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fine grid, derived mode only.

    Returns (tidy, play_level, signed_error). `tidy` matches Task B's
    aggregated schema (sigma, seed, velocity_source, predictor, role,
    step, seconds, mde, n, coverage) -- the CSV deliverable, covering
    every step, not just play_level_steps. `play_level` keeps individual
    plays separate (sigma, seed, step, game_id, play_id, predictor, mde
    -- constant_velocity and kalman only) at each step in
    `play_level_steps` -- the tidy table has already thrown play
    identity away by aggregating, but the play-level bootstrap needs it,
    and TIER1_SPEC_5.md needs it at more than one horizon (the honest
    primary horizon and the old 3s secondary). `signed_error` is one row
    per (sigma, seed, predictor, signed_error) for SIGNED_ERROR_PREDICTORS
    at 3s -- see run_tier1_sweep.compute_signed_error.
    """
    tidy_rows = []
    play_rows = []
    signed_rows = []
    combos = [(s, seed) for s in CROSSOVER_SIGMAS
             for seed in range(CROSSOVER_SEEDS)]
    for i, (sigma, seed) in enumerate(combos, 1):
        rng = np.random.default_rng(DEGRADE_BASE_SEED + seed)
        degraded_inp = corrupt_input(clean_inp, sigma, rng, "derived")
        samples = build_samples(degraded_inp, clean_out)

        set_filter_R(sigma, "derived")
        predictors = build_predictors(sigma_a)

        errors = evaluate(samples, predictors)
        tidy_rows.append(summarize_run(errors, sigma, seed, "derived"))

        for sample in samples:
            for step in play_level_steps:
                if sample.horizon < step:
                    continue
                for name in ("constant_velocity", "kalman"):
                    pred = predictors[name](sample, sample.horizon)
                    err = displacement_error(pred, sample.truth)[:, step - 1]
                    play_rows.append((sigma, seed, step, sample.game_id,
                                      sample.play_id, name, float(np.mean(err))))

        for name, value in compute_signed_error(
                samples, predictors, SIGNED_ERROR_PREDICTORS, STEP_3S):
            signed_rows.append((sigma, seed, name, value))

        print(f"  [{i}/{len(combos)}] sigma={sigma} seed={seed}: "
              f"{len(samples)} plays scored")

    tidy = pd.concat(tidy_rows, ignore_index=True)
    play_level = pd.DataFrame(play_rows, columns=[
        "sigma", "seed", "step", "game_id", "play_id", "predictor", "mde"])
    signed_error = pd.DataFrame(signed_rows, columns=[
        "sigma", "seed", "predictor", "signed_error"])
    return tidy, play_level, signed_error


# --------------------------------------------------------------- crossing

def find_crossing(sigma_grid, mde_kalman, mde_cv) -> float | None:
    """Linear interpolation of the sigma where kalman's 3s MDE crosses
    below constant_velocity's (diff = kalman - cv goes from + to -)."""
    diff = np.asarray(mde_kalman, dtype=float) - np.asarray(mde_cv, dtype=float)
    sigma_grid = np.asarray(sigma_grid, dtype=float)
    for i in range(len(sigma_grid) - 1):
        d0, d1 = diff[i], diff[i + 1]
        if np.isnan(d0) or np.isnan(d1):
            continue
        if d0 == 0:
            return float(sigma_grid[i])
        if np.sign(d0) != np.sign(d1):
            x0, x1 = sigma_grid[i], sigma_grid[i + 1]
            return float(x0 + (0 - d0) * (x1 - x0) / (d1 - d0))
    return None


def _play_sigma_matrix(play_level: pd.DataFrame, sigma_grid, predictor: str,
                       step: int) -> tuple[np.ndarray, list]:
    """(n_plays, n_sigma) matrix of per-play mean MDE at `step` (averaged
    over seeds), for one predictor -- the array the bootstrap resamples
    rows of."""
    per_play = (play_level[(play_level["predictor"] == predictor)
                          & (play_level["step"] == step)]
               .groupby(["game_id", "play_id", "sigma"])["mde"]
               .mean().reset_index())
    plays = sorted(per_play[["game_id", "play_id"]]
                   .drop_duplicates().itertuples(index=False, name=None))
    play_index = {p: i for i, p in enumerate(plays)}
    sigma_index = {s: i for i, s in enumerate(sigma_grid)}

    mat = np.full((len(plays), len(sigma_grid)), np.nan)
    for row in per_play.itertuples(index=False):
        mat[play_index[(row.game_id, row.play_id)], sigma_index[row.sigma]] = row.mde
    return mat, plays


def point_estimate_crossing(play_level: pd.DataFrame, sigma_grid,
                            step: int) -> float | None:
    mat_kalman, _ = _play_sigma_matrix(play_level, sigma_grid, "kalman", step)
    mat_cv, _ = _play_sigma_matrix(play_level, sigma_grid, "constant_velocity", step)
    return find_crossing(sigma_grid, np.nanmean(mat_kalman, axis=0),
                         np.nanmean(mat_cv, axis=0))


def bootstrap_crossing(play_level: pd.DataFrame, sigma_grid, step: int,
                       n_resamples: int = 1000, seed: int = 0
                       ) -> np.ndarray:
    """Play-level bootstrap: resample plays with replacement (all of a
    resampled play's seed-replicates come along with it), recompute the
    crossing sigma on each resample."""
    mat_kalman, plays = _play_sigma_matrix(play_level, sigma_grid, "kalman", step)
    mat_cv, _ = _play_sigma_matrix(play_level, sigma_grid, "constant_velocity", step)
    n_plays = len(plays)

    rng = np.random.default_rng(seed)
    crossings = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n_plays, size=n_plays)
        mean_kalman = np.nanmean(mat_kalman[idx], axis=0)
        mean_cv = np.nanmean(mat_cv[idx], axis=0)
        c = find_crossing(sigma_grid, mean_kalman, mean_cv)
        if c is not None:
            crossings.append(c)
    return np.array(crossings)


# ------------------------------------------------------------------ theory

def theory_table(degradation_csv, crossover_csv) -> pd.DataFrame:
    """MDE_noise ~= 1.253 * 3 * sqrt(2) * sigma / dt = 53.2 * sigma
    (sqrt(2) from differencing two independent position samples, /dt for
    the interval, x3 for the 3s extrapolation horizon, x1.253 for the
    mean of a 2-D Rayleigh displacement), combined in quadrature with the
    sigma=0 floor, for constant_velocity, derived mode, both grids.
    """
    coeff = 1.253 * 3.0 * np.sqrt(2) / DT

    frames = []
    for path in (degradation_csv, crossover_csv):
        df = pd.read_csv(path)
        sub = df[(df["predictor"] == "constant_velocity")
                & (df["velocity_source"] == "derived")
                & (df["step"] == STEP_3S)]
        per_run = (sub.groupby(["sigma", "seed"])
                  .apply(lambda g: np.average(g["mde"], weights=g["n"]),
                         include_groups=False)
                  .rename("mde").reset_index())
        frames.append(per_run.groupby("sigma")["mde"].mean())
    measured = pd.concat(frames).groupby(level=0).mean().sort_index()

    floor = measured.loc[0.0]
    sigmas = measured.index.to_numpy()
    predicted = np.sqrt(floor**2 + (coeff * sigmas)**2)

    table = pd.DataFrame({
        "sigma": sigmas,
        "predicted": predicted,
        "measured": measured.to_numpy(),
    })
    table["ratio"] = table["measured"] / table["predicted"]
    return table


# ----------------------------------------------------------------- figures

def plot_crossover(play_level, tidy, crossing_sigma, crossing_ci, path):
    import matplotlib.pyplot as plt

    at3s = tidy[tidy["step"] == STEP_3S]
    per_run = (at3s.groupby(["sigma", "seed", "predictor"])
              .apply(lambda g: np.average(g["mde"], weights=g["n"]),
                     include_groups=False)
              .rename("mde").reset_index())
    summary = (per_run.groupby(["sigma", "predictor"])["mde"]
              .agg(mean="mean", std="std").reset_index())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axvspan(*NGS_SPEC_RANGE, color="0.85", zorder=0,
              label=f"NGS spec range ({NGS_SPEC_RANGE[0]}-{NGS_SPEC_RANGE[1]} yd)")

    for name in MAIN_FIGURE_PREDICTORS:
        grp = summary[summary["predictor"] == name].sort_values("sigma")
        std = grp["std"].fillna(0)
        ax.plot(grp["sigma"], grp["mean"], marker="o", linewidth=1.8, label=name)
        ax.fill_between(grp["sigma"], grp["mean"] - std, grp["mean"] + std,
                        alpha=0.15)

    if crossing_sigma is not None:
        ax.axvline(crossing_sigma, color="0.3", linestyle="--", linewidth=1.2)
        label = f"crossing @ sigma={crossing_sigma:.3f}"
        if crossing_ci is not None:
            label += f" [{crossing_ci[0]:.3f}, {crossing_ci[1]:.3f}]"
            ax.axvspan(crossing_ci[0], crossing_ci[1], color="tab:red", alpha=0.08)
        ax.text(crossing_sigma, 39, label,
               rotation=90, va="top", ha="right", fontsize=8, color="0.3")

    ax.set_xlim(0, 0.30)
    ax.set_ylim(5, 40)
    ax.set_xlabel("position noise sigma (yd)")
    ax.set_ylabel("3 s mean displacement error (yd)")
    ax.set_title("Crossover region: constant_velocity vs. state estimation "
                 "(derived mode)")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def plot_robustness_full(tidy_task_b, path):
    from run_tier1_sweep import plot_robustness

    caption = (
        "Full 0-5 yd range, log y, every predictor including "
        "constant_acceleration's double-differencing blow-up. "
        "constant_acceleration, damped_velocity and constant_position are "
        "excluded from the main crossover figure (tier1_crossover.png) "
        "because they are never competitive at any sigma tested and would "
        "dominate a linear-scale plot of the region that matters.")
    plot_robustness(tidy_task_b, path, caption=caption)


# --------------------------------------------------------------------- main

def main():
    print("loading clean dataset...")
    clean_inp, clean_out = load_clean()
    sigma_a = frozen_sigma_a(clean_inp, clean_out)

    n_combos = len(CROSSOVER_SIGMAS) * CROSSOVER_SEEDS
    print(f"\nrunning crossover sweep: {len(CROSSOVER_SIGMAS)} sigmas x "
         f"{CROSSOVER_SEEDS} seeds = {n_combos} runs, derived mode only")
    tidy, play_level, signed_error = run_crossover_sweep(
        sigma_a, clean_inp, clean_out)

    tidy.to_csv(OUTPUT_DIR / "tier1_crossover.csv", index=False)
    print(f"\nwrote {OUTPUT_DIR / 'tier1_crossover.csv'} ({len(tidy):,} rows)")

    signed_error.to_csv(OUTPUT_DIR / "tier1_signed_error_crossover.csv",
                        index=False)
    print(f"wrote {OUTPUT_DIR / 'tier1_signed_error_crossover.csv'} "
         f"({len(signed_error):,} rows)")

    print("\nmean signed along-track error at 3s, derived mode "
         "(negative = undershoot):")
    print(signed_error.groupby(["sigma", "predictor"])["signed_error"]
         .mean().unstack("predictor").round(3).to_string())

    point = point_estimate_crossing(play_level, CROSSOVER_SIGMAS, STEP_3S)
    print(f"\npoint-estimate crossing sigma (kalman vs constant_velocity, "
         f"3s, derived): {point}")

    print("running play-level bootstrap (1000 resamples)...")
    boot = bootstrap_crossing(play_level, CROSSOVER_SIGMAS, STEP_3S, n_resamples=1000)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"bootstrap: {len(boot)}/1000 resamples found a crossing")
    print(f"95% interval: [{lo:.4f}, {hi:.4f}] yd  (width {hi - lo:.4f})")
    if (hi - lo) > 0.06:
        print("  interval is wider than ~+/-0.03 yd -- treat the point "
             "estimate as approximate; more plays would narrow it faster "
             "than more seeds, since the bootstrap resamples plays")

    plot_crossover(play_level, tidy, point, (lo, hi),
                  OUTPUT_DIR / "tier1_crossover.png")

    print("\ntoward_ball_land robustness check, 3s MDE at each sigma "
         "(derived mode):")
    at3s = tidy[tidy["step"] == STEP_3S]
    per_run = (at3s.groupby(["sigma", "predictor"])
              .apply(lambda g: np.average(g["mde"], weights=g["n"]),
                     include_groups=False)
              .rename("mde").reset_index())
    pivot = per_run.pivot(index="predictor", columns="sigma", values="mde")
    print(pivot.loc[["toward_ball_land", "kalman", "constant_velocity",
                    "velocity_turning_to_ball", "imm", "kalman_velocity"]]
         .round(2).to_string())

    print("\ntheory cross-check (constant_velocity, derived, both grids):")
    table = theory_table(
        "outputs/tier1_degradation.csv", OUTPUT_DIR / "tier1_crossover.csv")
    print(table.round(4).to_string(index=False))

    task_b = pd.read_csv("outputs/tier1_degradation.csv")
    plot_robustness_full(task_b, OUTPUT_DIR / "tier1_robustness_full.png")


if __name__ == "__main__":
    main()
