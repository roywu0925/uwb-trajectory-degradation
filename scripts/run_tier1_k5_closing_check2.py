"""Item 4(b) extended: distance to toward_ball_land's prediction, K3 vs
K5, across the sigma_player grid -- the specific testable prediction
("gets closer as sigma rises") that the clean-data-only check couldn't
address. Forward-pass only, fold-0 checkpoints.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from baselines import BASELINES
from config import OUTPUT_DIR, PLAYER_SIGMAS, WEEKS
from data_loader import build_samples
from degrade import corrupt_input
from model import GRUResidualModel, build_examples
from run_tier1_taskk import build_tensors, predict_all
from run_tier1_taskm import load_train_val_raw

PRIMARY_STEP = 19
VELOCITY_SOURCE = "derived"
DEGRADE_BASE_SEED_M = 20260929


def load_ckpt(variant, fold_id=0):
    ckpt = torch.load(OUTPUT_DIR / "checkpoints" / f"{variant}_fold{fold_id}.pt", weights_only=False)
    model = GRUResidualModel(hidden=ckpt["config"]["hidden"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)

    clean_inp, clean_out = load_train_val_raw(train_val_games)
    k3 = load_ckpt("K3", 0)
    k5 = load_ckpt("K5", 0)
    col = PRIMARY_STEP - 1

    print(f"{'sigma':>6s} {'K3<->tbl':>10s} {'K5<->tbl':>10s} {'K3 err':>8s} {'K5 err':>8s}")
    for sigma in PLAYER_SIGMAS:
        rng = np.random.default_rng(DEGRADE_BASE_SEED_M)
        degraded_inp = corrupt_input(clean_inp, sigma, rng, VELOCITY_SOURCE)
        degraded_samples = build_samples(degraded_inp, clean_out)

        examples = build_examples(degraded_samples)
        tensors = build_tensors(examples)
        game_id_arr = tensors["game_id"]
        test_idx = np.where(np.isin(game_id_arr, folds[0]))[0]
        anchor_np = tensors["anchor"].numpy()
        base_anchor = np.repeat(anchor_np[:, None, :], tensors["truth"].shape[1], axis=1)
        mask_np = tensors["mask"].numpy()
        truth_np = tensors["truth"].numpy()

        tbl_preds = np.zeros_like(base_anchor)
        pos = 0
        for sample in degraded_samples:
            n = len(sample.state)
            pred = BASELINES["toward_ball_land"](sample, tensors["truth"].shape[1])
            tbl_preds[pos:pos+n] = pred
            pos += n

        out3 = predict_all(k3, torch.from_numpy(base_anchor), tensors, test_idx)
        pred3 = base_anchor[test_idx] + out3
        out5 = predict_all(k5, torch.from_numpy(base_anchor), tensors, test_idx)
        pred5 = base_anchor[test_idx] + out5

        valid = mask_np[test_idx, col] > 0
        d3 = np.linalg.norm(pred3[valid, col] - tbl_preds[test_idx][valid, col], axis=-1).mean()
        d5 = np.linalg.norm(pred5[valid, col] - tbl_preds[test_idx][valid, col], axis=-1).mean()
        e3 = np.linalg.norm(pred3[valid, col] - truth_np[test_idx][valid, col], axis=-1).mean()
        e5 = np.linalg.norm(pred5[valid, col] - truth_np[test_idx][valid, col], axis=-1).mean()
        print(f"{sigma:6.1f} {d3:10.4f} {d5:10.4f} {e3:8.4f} {e5:8.4f}")


if __name__ == "__main__":
    main()
