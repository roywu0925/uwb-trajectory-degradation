"""Item 1: the K5 / tbl_kalman_speed crossing sigma, fine grid between
1.0 and 2.0, play-level bootstrap -- same method as Part 1's crossovers
and the K3/K4 crossover script. Adaptive R (the headline convention),
not frozen -- this is the main-line comparison, not the fairness
diagnostic.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from config import OUTPUT_DIR, WEEKS
from data_loader import build_samples
from degrade import corrupt_input
from kalman import calibrate_sigma_a, estimate_sigma_a
from model import GRUResidualModel, build_examples
from run_tier1_crossover import find_crossing
import run_tier1_crossover as co
from run_tier1_taskk import build_tensors, predict_all
from run_tier1_taskm import load_train_val_raw
from run_tier1_taskp import build_predictors_quadrature
from run_tier1_r_convention_check import set_filter_R_convention

PRIMARY_STEP = 19
VELOCITY_SOURCE = "derived"
DEGRADE_BASE_SEED_M = 20260929
SIGMA_GRID = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
SEEDS = [0, 1, 2]


def load_checkpoints(folds):
    ckpts = {}
    for i in range(len(folds)):
        ckpt = torch.load(OUTPUT_DIR / "checkpoints" / f"K5_fold{i}.pt", weights_only=False)
        model = GRUResidualModel(hidden=ckpt["config"]["hidden"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        ckpts[i] = model
    return ckpts


def main():
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)

    clean_inp, clean_out = load_train_val_raw(train_val_games)
    clean_samples_for_sigma = build_samples(clean_inp, clean_out)
    sigma_a_raw = estimate_sigma_a(clean_samples_for_sigma)
    sigma_a = calibrate_sigma_a(clean_samples_for_sigma, sigma_a_hi=sigma_a_raw, target=2.0)

    ck5 = load_checkpoints(folds)

    play_rows = []
    combos = [(s, seed) for s in SIGMA_GRID for seed in SEEDS]
    for k, (sigma, seed) in enumerate(combos, 1):
        rng = np.random.default_rng(DEGRADE_BASE_SEED_M + seed)
        degraded_inp = corrupt_input(clean_inp, sigma, rng, VELOCITY_SOURCE)
        degraded_samples = build_samples(degraded_inp, clean_out)

        examples = build_examples(degraded_samples)
        tensors = build_tensors(examples)
        anchor_np = tensors["anchor"].numpy()
        base5 = np.repeat(anchor_np[:, None, :], tensors["truth"].shape[1], axis=1)
        game_id_arr = tensors["game_id"]
        meta = [(e.game_id, e.play_id) for e in examples]

        set_filter_R_convention(sigma, VELOCITY_SOURCE, "quadrature")
        predictors = build_predictors_quadrature(sigma_a)
        tks_pred_fn = predictors["tbl_kalman_speed"]

        mask_np = tensors["mask"].numpy()
        truth_np = tensors["truth"].numpy()
        col = PRIMARY_STEP - 1

        for i, fold_games in enumerate(folds):
            idx = np.where(np.isin(game_id_arr, fold_games))[0]
            out5 = predict_all(ck5[i], torch.from_numpy(base5), tensors, idx)
            pred5 = base5[idx] + out5
            for local_j, global_j in enumerate(idx):
                if mask_np[global_j, col] <= 0:
                    continue
                err = float(np.linalg.norm(pred5[local_j, col] - truth_np[global_j, col]))
                gid, pid = meta[global_j]
                play_rows.append((sigma, seed, PRIMARY_STEP, gid, pid, "K5", err))

        pos = 0
        for sample in degraded_samples:
            n = len(sample.state)
            pred_tks = tks_pred_fn(sample, tensors["truth"].shape[1])
            for j in range(n):
                global_j = pos + j
                if mask_np[global_j, col] <= 0:
                    continue
                err = float(np.linalg.norm(pred_tks[j, col] - truth_np[global_j, col]))
                play_rows.append((sigma, seed, PRIMARY_STEP, sample.game_id, sample.play_id,
                                  "tbl_kalman_speed", err))
            pos += n

        print(f"[{k}/{len(combos)}] sigma={sigma} seed={seed} done")

    play_level = pd.DataFrame(play_rows, columns=[
        "sigma", "seed", "step", "game_id", "play_id", "predictor", "mde"])
    play_level.to_csv(OUTPUT_DIR / "tier1_k5_tks_crossover_playlevel.csv", index=False)
    per_play = (play_level.groupby(["sigma", "seed", "step", "game_id", "play_id", "predictor"])
               ["mde"].mean().reset_index())

    mat_k5, plays = co._play_sigma_matrix(per_play, SIGMA_GRID, "K5", PRIMARY_STEP)
    mat_tks, _ = co._play_sigma_matrix(per_play, SIGMA_GRID, "tbl_kalman_speed", PRIMARY_STEP)

    point = find_crossing(SIGMA_GRID, np.nanmean(mat_k5, axis=0), np.nanmean(mat_tks, axis=0))
    print(f"\npoint-estimate K5/tbl_kalman_speed crossing sigma: {point}")

    rng = np.random.default_rng(0)
    n_plays = len(plays)
    crossings = []
    for _ in range(1000):
        idx = rng.integers(0, n_plays, size=n_plays)
        mean_k5 = np.nanmean(mat_k5[idx], axis=0)
        mean_tks = np.nanmean(mat_tks[idx], axis=0)
        c = find_crossing(SIGMA_GRID, mean_k5, mean_tks)
        if c is not None:
            crossings.append(c)
    crossings = np.array(crossings)
    print(f"bootstrap (n={len(crossings)}/1000 valid): "
         f"median={np.median(crossings):.4f} "
         f"95% CI=({np.percentile(crossings,2.5):.4f}, {np.percentile(crossings,97.5):.4f})")
    print(f"\nK5 mean mde by sigma: {np.round(np.nanmean(mat_k5, axis=0), 4)}")
    print(f"tbl_kalman_speed mean mde by sigma: {np.round(np.nanmean(mat_tks, axis=0), 4)}")
    print(f"sigma grid: {SIGMA_GRID}")


if __name__ == "__main__":
    main()
