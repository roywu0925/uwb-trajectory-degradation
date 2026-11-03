"""Gain measurement, extended: K1, K3 (standard features) and K4
(Kalman-filtered velocity features), mean/median/p90, anchor-only
perturbation (the direct analog of constant_velocity's own mechanism --
compare against the physical integration gain of 1.9 at this horizon).
Forward-pass only, clean data, fold-0 checkpoints.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from cache_pipeline import build_dataset_cached
from config import OUTPUT_DIR, WEEKS
from kalman import build_Q, calibrate_sigma_a, estimate_sigma_a
from model import GRUResidualModel, build_examples, build_examples_k4
from run_tier1_taskk import build_tensors

PRIMARY_STEP = 19
DELTA = 0.01


def gain_for_variant(variant, examples, tensors, test_idx):
    ckpt = torch.load(OUTPUT_DIR / "checkpoints" / f"{variant}_fold0.pt", weights_only=False)
    model = GRUResidualModel(hidden=ckpt["config"]["hidden"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    hist_padded = tensors["hist_padded"].numpy()
    hist_lengths = tensors["hist_lengths"].numpy()
    static_feat = tensors["static_feat"]

    gains_x, gains_y = [], []
    with torch.no_grad():
        for idx in test_idx:
            L = hist_lengths[idx]
            base_seq = hist_padded[idx:idx+1, :L, :].copy()

            def forward(seq):
                hp = torch.from_numpy(hist_padded[idx:idx+1].copy())
                hp[:, :L, :] = torch.from_numpy(seq)
                lengths = torch.tensor([L])
                static = static_feat[idx:idx+1]
                out = model(hp, lengths, static)
                return out[0, PRIMARY_STEP - 1].numpy()

            out0 = forward(base_seq[0])
            seq_x = base_seq.copy()
            seq_y = base_seq.copy()
            seq_x[0, -1, 2] += DELTA
            seq_y[0, -1, 3] += DELTA
            out_x = forward(seq_x[0])
            out_y = forward(seq_y[0])
            gains_x.append((out_x[0] - out0[0]) / DELTA)
            gains_y.append((out_y[1] - out0[1]) / DELTA)

    gains_x = np.array(gains_x)
    gains_y = np.array(gains_y)
    return gains_x, gains_y


def main():
    samples, _ = build_dataset_cached(WEEKS)
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)
    train_val_samples = [s for s in samples if s.game_id in train_val_games]

    # K1/K3: standard features
    examples_std = build_examples(train_val_samples)
    tensors_std = build_tensors(examples_std)
    test_idx_std = np.where(np.isin(tensors_std["game_id"], folds[0]))[0]

    # K4: Kalman-filtered velocity features
    sigma_a_raw = estimate_sigma_a(train_val_samples)
    sigma_a = calibrate_sigma_a(train_val_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    Q = build_Q(sigma_a)
    examples_k4 = build_examples_k4(train_val_samples, Q)
    tensors_k4 = build_tensors(examples_k4)
    test_idx_k4 = np.where(np.isin(tensors_k4["game_id"], folds[0]))[0]

    print(f"{'variant':6s} {'axis':6s} {'mean':>8s} {'median':>8s} {'p90':>8s}")
    for variant, examples, tensors, test_idx in (
        ("K1", examples_std, tensors_std, test_idx_std),
        ("K3", examples_std, tensors_std, test_idx_std),
        ("K4", examples_k4, tensors_k4, test_idx_k4),
    ):
        gx, gy = gain_for_variant(variant, examples, tensors, test_idx)
        print(f"{variant:6s} {'vx->x':6s} {gx.mean():8.3f} {np.median(gx):8.3f} {np.percentile(gx,90):8.3f}")
        print(f"{variant:6s} {'vy->y':6s} {gy.mean():8.3f} {np.median(gy):8.3f} {np.percentile(gy,90):8.3f}")
    print(f"\nphysical integration gain at 1.9s horizon = 1.9")


if __name__ == "__main__":
    main()
