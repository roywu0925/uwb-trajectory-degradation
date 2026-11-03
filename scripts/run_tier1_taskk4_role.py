"""K4 per-role out-of-fold breakdown, clean data, all 5 folds combined --
forward-pass only, reusing the already-trained K4_fold{i}.pt checkpoints.
Matches Task L's K1/K2/K3 per-role table format exactly.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from cache_pipeline import build_dataset_cached
from config import OUTPUT_DIR, WEEKS
from kalman import build_Q, calibrate_sigma_a, estimate_sigma_a
from model import GRUResidualModel, build_examples_k4
from run_tier1_taskk import build_tensors, predict_all

PRIMARY_STEP = 19


def masked_error_by_role(pred, truth, mask, roles, step):
    col = step - 1
    valid = mask[:, col] > 0
    err = np.linalg.norm(pred[valid, col] - truth[valid, col], axis=-1)
    r = roles[valid]
    out = {}
    for role in np.unique(r):
        m = r == role
        out[role] = (float(err[m].mean()), int(m.sum()))
    out["overall"] = (float(err.mean()), int(valid.sum()))
    return out


def main():
    samples, _ = build_dataset_cached(WEEKS)
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)
    train_val_samples = [s for s in samples if s.game_id in train_val_games]

    sigma_a_raw = estimate_sigma_a(train_val_samples)
    sigma_a = calibrate_sigma_a(train_val_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    Q = build_Q(sigma_a)

    examples = build_examples_k4(train_val_samples, Q)
    roles = np.array([e.role for e in examples])
    tensors = build_tensors(examples)
    truth_np = tensors["truth"].numpy()
    mask_np = tensors["mask"].numpy()
    game_id_arr = tensors["game_id"]
    anchor_np = tensors["anchor"].numpy()
    base_anchor_repeat = np.repeat(anchor_np[:, None, :], truth_np.shape[1], axis=1)
    base_t = torch.from_numpy(base_anchor_repeat)

    n = len(examples)
    oof_pred = np.full((n, truth_np.shape[1], 2), np.nan, dtype=np.float32)
    for i, fold_games in enumerate(folds):
        ckpt = torch.load(OUTPUT_DIR / "checkpoints" / f"K4_fold{i}.pt", weights_only=False)
        model = GRUResidualModel(hidden=ckpt["config"]["hidden"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        test_idx = np.where(np.isin(game_id_arr, fold_games))[0]
        model_out = predict_all(model, base_t, tensors, test_idx)
        oof_pred[test_idx] = base_anchor_repeat[test_idx] + model_out

    by_role = masked_error_by_role(oof_pred, truth_np, mask_np, roles, PRIMARY_STEP)
    print("K4 -- out-of-fold, by role (all 5 folds combined), @1.9s")
    for k, (e, n_v) in by_role.items():
        print(f"  {k}: err={e:.4f} n={n_v}")


if __name__ == "__main__":
    main()
