"""Item 2: the K3/K4 crossing sigma, with a play-level bootstrap interval
-- the direct analogue of Part 1's constant_velocity/kalman crossing
(run_tier1_crossover.py), one layer up in the learned half.

Fine grid (config.CROSSOVER_SIGMAS, the same grid Part 1 used to locate
its crossing), CV folds only, forward-pass only against the already
trained K3/K4 checkpoints -- no retraining.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from config import CROSSOVER_SIGMAS, OUTPUT_DIR, WEEKS
from data_loader import build_samples
from degrade import corrupt_input
from kalman import build_Q, calibrate_sigma_a, estimate_sigma_a
from model import GRUResidualModel, build_examples, build_examples_k4
from run_tier1_crossover import bootstrap_crossing, find_crossing, point_estimate_crossing
from run_tier1_taskk import build_tensors, predict_all
from run_tier1_taskm import load_train_val_raw

PRIMARY_STEP = 19
VELOCITY_SOURCE = "derived"
DEGRADE_BASE_SEED_M = 20260929
SEEDS = [0, 1, 2]


def load_checkpoints(folds, variant):
    ckpts = {}
    for i in range(len(folds)):
        ckpt = torch.load(OUTPUT_DIR / "checkpoints" / f"{variant}_fold{i}.pt", weights_only=False)
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
    Q = build_Q(sigma_a)

    ck3 = load_checkpoints(folds, "K3")
    ck4 = load_checkpoints(folds, "K4")

    combos = [(s, seed) for s in CROSSOVER_SIGMAS for seed in (SEEDS if s > 0 else [0])]

    play_rows = []
    for k, (sigma, seed) in enumerate(combos, 1):
        rng = np.random.default_rng(DEGRADE_BASE_SEED_M + seed)
        degraded_inp = corrupt_input(clean_inp, sigma, rng, VELOCITY_SOURCE)
        degraded_samples = build_samples(degraded_inp, clean_out)

        examples_k3 = build_examples(degraded_samples)
        tensors_k3 = build_tensors(examples_k3)
        anchor3 = tensors_k3["anchor"].numpy()
        base3 = np.repeat(anchor3[:, None, :], tensors_k3["truth"].shape[1], axis=1)
        game_id_k3 = tensors_k3["game_id"]

        examples_k4 = build_examples_k4(degraded_samples, Q)
        tensors_k4 = build_tensors(examples_k4)
        anchor4 = tensors_k4["anchor"].numpy()
        base4 = np.repeat(anchor4[:, None, :], tensors_k4["truth"].shape[1], axis=1)
        game_id_k4 = tensors_k4["game_id"]

        meta3 = [(e.game_id, e.play_id) for e in examples_k3]
        meta4 = [(e.game_id, e.play_id) for e in examples_k4]

        for i, fold_games in enumerate(folds):
            idx3 = np.where(np.isin(game_id_k3, fold_games))[0]
            idx4 = np.where(np.isin(game_id_k4, fold_games))[0]

            out3 = predict_all(ck3[i], torch.from_numpy(base3), tensors_k3, idx3)
            pred3 = base3[idx3] + out3
            mask3 = tensors_k3["mask"].numpy()[idx3]
            truth3 = tensors_k3["truth"].numpy()[idx3]

            out4 = predict_all(ck4[i], torch.from_numpy(base4), tensors_k4, idx4)
            pred4 = base4[idx4] + out4
            mask4 = tensors_k4["mask"].numpy()[idx4]
            truth4 = tensors_k4["truth"].numpy()[idx4]

            col = PRIMARY_STEP - 1
            for local_j, global_j in enumerate(idx3):
                if mask3[local_j, col] <= 0:
                    continue
                err = float(np.linalg.norm(pred3[local_j, col] - truth3[local_j, col]))
                gid, pid = meta3[global_j]
                play_rows.append((sigma, seed, PRIMARY_STEP, gid, pid, "K3", err))
            for local_j, global_j in enumerate(idx4):
                if mask4[local_j, col] <= 0:
                    continue
                err = float(np.linalg.norm(pred4[local_j, col] - truth4[local_j, col]))
                gid, pid = meta4[global_j]
                play_rows.append((sigma, seed, PRIMARY_STEP, gid, pid, "K4", err))

        print(f"[{k}/{len(combos)}] sigma={sigma} seed={seed} done")

    play_level = pd.DataFrame(play_rows, columns=[
        "sigma", "seed", "step", "game_id", "play_id", "predictor", "mde"])
    play_level.to_csv(OUTPUT_DIR / "tier1_k3k4_crossover_playlevel.csv", index=False)

    # mde here is per-TRACK, not yet per-play -- average within play first
    per_play = (play_level.groupby(["sigma", "seed", "step", "game_id", "play_id", "predictor"])
               ["mde"].mean().reset_index())

    # reuse run_tier1_crossover's play-sigma-matrix helper directly --
    # it only keys off predictor/step/game_id/play_id/sigma/mde columns,
    # all present here with the same names.
    import run_tier1_crossover as co

    def mat_for(predictor):
        return co._play_sigma_matrix(per_play, CROSSOVER_SIGMAS, predictor, PRIMARY_STEP)

    mat_k3, plays = mat_for("K3")
    mat_k4, _ = mat_for("K4")

    point = find_crossing(CROSSOVER_SIGMAS, np.nanmean(mat_k3, axis=0), np.nanmean(mat_k4, axis=0))
    print(f"\npoint-estimate K3/K4 crossing sigma: {point}")

    rng = np.random.default_rng(0)
    n_plays = len(plays)
    crossings = []
    for _ in range(1000):
        idx = rng.integers(0, n_plays, size=n_plays)
        mean_k3 = np.nanmean(mat_k3[idx], axis=0)
        mean_k4 = np.nanmean(mat_k4[idx], axis=0)
        c = find_crossing(CROSSOVER_SIGMAS, mean_k3, mean_k4)
        if c is not None:
            crossings.append(c)
    crossings = np.array(crossings)
    print(f"bootstrap (n={len(crossings)}/1000 valid): "
         f"median={np.median(crossings):.4f} "
         f"95% CI=({np.percentile(crossings,2.5):.4f}, {np.percentile(crossings,97.5):.4f})")

    print(f"\nK3 mean mde by sigma: {np.round(np.nanmean(mat_k3, axis=0), 4)}")
    print(f"K4 mean mde by sigma: {np.round(np.nanmean(mat_k4, axis=0), 4)}")
    print(f"sigma grid: {CROSSOVER_SIGMAS}")


if __name__ == "__main__":
    main()
