"""K5's sigma_player degradation sweep, matching run_tier1_taskm.py's
methodology exactly, so K1-K5 are comparable on the same axis. K5 uses
standard build_examples (same features as K1/K2/K3) and the
K5_fold{i}.pt checkpoints (trained with on-the-fly noise augmentation
at sigma=0.2) just trained.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from config import OUTPUT_DIR, PLAYER_SIGMAS, WEEKS
from data_loader import build_samples
from degrade import corrupt_input
from kalman import calibrate_sigma_a, estimate_sigma_a
from model import GRUResidualModel, build_examples
from run_tier1_taskk import build_tensors, masked_error_at_step, predict_all
from run_tier1_taskm import load_train_val_raw

PRIMARY_STEP = 19
VELOCITY_SOURCE = "derived"
GRID_SEEDS = 3
DEGRADE_BASE_SEED_M = 20260929


def main():
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)

    clean_inp, clean_out = load_train_val_raw(train_val_games)

    checkpoints = {}
    for i in range(len(folds)):
        ckpt = torch.load(OUTPUT_DIR / "checkpoints" / f"K5_fold{i}.pt", weights_only=False)
        model = GRUResidualModel(hidden=ckpt["config"]["hidden"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        checkpoints[i] = model

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
        game_id_arr = tensors["game_id"]
        anchor_np = tensors["anchor"].numpy()
        base_anchor = np.repeat(anchor_np[:, None, :], tensors["truth"].shape[1], axis=1)
        base_t = torch.from_numpy(base_anchor)

        truth_np = tensors["truth"].numpy()
        mask_np = tensors["mask"].numpy()
        n = len(examples)
        oof_pred = np.full((n, truth_np.shape[1], 2), np.nan, dtype=np.float32)
        for i, fold_games in enumerate(folds):
            test_idx = np.where(np.isin(game_id_arr, fold_games))[0]
            model_out = predict_all(checkpoints[i], base_t, tensors, test_idx)
            oof_pred[test_idx] = base_anchor[test_idx] + model_out

        err, n_v = masked_error_at_step(oof_pred, truth_np, mask_np, PRIMARY_STEP)
        all_rows.append({"sigma": sigma, "seed": seed, "predictor": "K5", "error": err, "n": n_v})
        print(f"  K5: {err:.4f} (n={n_v})")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "tier1_k5_degrade_raw.csv", index=False)
    summary = df.groupby("sigma")["error"].mean().round(4)
    print("\nK5 mean error @1.9s vs sigma_player:")
    print(summary.to_string())

    taskm_path = OUTPUT_DIR / "tier1_taskm_summary.csv"
    taskm = pd.read_csv(taskm_path, index_col=0)
    taskm["K5"] = summary
    taskm.to_csv(taskm_path)
    print("\nK1-K5 + baselines, combined:")
    print(taskm.to_string())


if __name__ == "__main__":
    main()
