"""Where does tbl_kalman_speed cross toward_ball_land (not constant_velocity)?
Baseline-only, no training. Same fine-grid + play-level-bootstrap method as
every other crossing in this project. Train+val population, matching the
project's standing crossover convention.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from baselines import BASELINES
from config import CROSSOVER_SIGMAS, OUTPUT_DIR, WEEKS
from data_loader import build_samples
from degrade import corrupt_input
from kalman import build_Q, calibrate_sigma_a, estimate_sigma_a
from run_tier1_crossover import find_crossing
import run_tier1_crossover as co
from run_tier1_taskm import load_train_val_raw
from run_tier1_taskp import build_predictors_quadrature
from run_tier1_r_convention_check import set_filter_R_convention

PRIMARY_STEP = 19
VELOCITY_SOURCE = "derived"
DEGRADE_BASE_SEED_M = 20260929
SEEDS = [0, 1, 2]
# extend past the standard crossover grid in case the crossing sits later
# than 0.30 -- tbl_kalman_speed starts further behind toward_ball_land
# (0.77 yd) than it does behind constant_velocity (0.68 yd)
EXTENDED_GRID = sorted(set(CROSSOVER_SIGMAS + [0.4, 0.5, 0.6, 0.8, 1.0]))


def main():
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)

    clean_inp, clean_out = load_train_val_raw(train_val_games)
    clean_samples_for_sigma = build_samples(clean_inp, clean_out)
    sigma_a_raw = estimate_sigma_a(clean_samples_for_sigma)
    sigma_a = calibrate_sigma_a(clean_samples_for_sigma, sigma_a_hi=sigma_a_raw, target=2.0)
    Q = build_Q(sigma_a)

    play_rows = []
    combos = [(s, seed) for s in EXTENDED_GRID for seed in (SEEDS if s > 0 else [0])]
    for k, (sigma, seed) in enumerate(combos, 1):
        rng = np.random.default_rng(DEGRADE_BASE_SEED_M + seed)
        degraded_inp = corrupt_input(clean_inp, sigma, rng, VELOCITY_SOURCE)
        degraded_samples = build_samples(degraded_inp, clean_out)

        set_filter_R_convention(sigma, VELOCITY_SOURCE, "quadrature")
        predictors = build_predictors_quadrature(sigma_a)
        col = PRIMARY_STEP - 1

        for name, fn in (("toward_ball_land", BASELINES["toward_ball_land"]),
                         ("tbl_kalman_speed", predictors["tbl_kalman_speed"])):
            for sample in degraded_samples:
                if sample.horizon < PRIMARY_STEP:
                    continue
                pred = fn(sample, sample.horizon)
                err = np.linalg.norm(pred[:, col] - sample.truth[:, col], axis=-1).mean()
                play_rows.append((sigma, seed, PRIMARY_STEP, sample.game_id, sample.play_id,
                                  name, float(err)))

        print(f"[{k}/{len(combos)}] sigma={sigma} seed={seed} done")

    play_level = pd.DataFrame(play_rows, columns=[
        "sigma", "seed", "step", "game_id", "play_id", "predictor", "mde"])
    play_level.to_csv(OUTPUT_DIR / "tier1_tbl_crossing_playlevel.csv", index=False)
    per_play = (play_level.groupby(["sigma", "seed", "step", "game_id", "play_id", "predictor"])
               ["mde"].mean().reset_index())

    mat_tbl, plays = co._play_sigma_matrix(per_play, EXTENDED_GRID, "toward_ball_land", PRIMARY_STEP)
    mat_tks, _ = co._play_sigma_matrix(per_play, EXTENDED_GRID, "tbl_kalman_speed", PRIMARY_STEP)

    print(f"\ntoward_ball_land mean mde by sigma: {np.round(np.nanmean(mat_tbl, axis=0), 4)}")
    print(f"tbl_kalman_speed mean mde by sigma:  {np.round(np.nanmean(mat_tks, axis=0), 4)}")
    print(f"grid: {EXTENDED_GRID}")

    point = find_crossing(EXTENDED_GRID, np.nanmean(mat_tks, axis=0), np.nanmean(mat_tbl, axis=0))
    print(f"\npoint-estimate tbl_kalman_speed/toward_ball_land crossing sigma: {point}")

    if point is None:
        print("NO CROSSING FOUND in this grid range -- tbl_kalman_speed does not "
             "overtake toward_ball_land anywhere tested.")
        return

    rng = np.random.default_rng(0)
    n_plays = len(plays)
    crossings = []
    for _ in range(1000):
        idx = rng.integers(0, n_plays, size=n_plays)
        mean_tbl = np.nanmean(mat_tbl[idx], axis=0)
        mean_tks = np.nanmean(mat_tks[idx], axis=0)
        c = find_crossing(EXTENDED_GRID, mean_tks, mean_tbl)
        if c is not None:
            crossings.append(c)
    crossings = np.array(crossings)
    print(f"bootstrap (n={len(crossings)}/1000 valid): "
         f"median={np.median(crossings):.4f} "
         f"95% CI=({np.percentile(crossings,2.5):.4f}, {np.percentile(crossings,97.5):.4f})")


if __name__ == "__main__":
    main()
