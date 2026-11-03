"""Item 4, the closing check: has K5 independently converged on
toward_ball_land's design (geometry from the ~25yd ball_land baseline,
not the 0.1s velocity reading)? Three forward-pass-only tests, K3 vs K4
vs K5 (or K3 vs K5 where the hypothesis is specifically about K5).
Reports what's actually found, not fit to the hypothesis.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from baselines import BASELINES
from cache_pipeline import build_dataset_cached
from config import OUTPUT_DIR, WEEKS
from kalman import build_Q, calibrate_sigma_a, estimate_sigma_a
from model import GRUResidualModel, build_examples, build_examples_k4
from run_tier1_taskk import build_tensors, predict_all

PRIMARY_STEP = 19
DELTA = 0.01


def load_ckpt(variant, fold_id=0):
    ckpt = torch.load(OUTPUT_DIR / "checkpoints" / f"{variant}_fold{fold_id}.pt", weights_only=False)
    model = GRUResidualModel(hidden=ckpt["config"]["hidden"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def ball_land_gain(model, tensors, test_idx):
    """d(output)/d(ball_land_offset), same finite-difference recipe as
    the velocity gain test, perturbing static_feat's last 2 columns."""
    hist_padded = tensors["hist_padded"]
    hist_lengths = tensors["hist_lengths"]
    static_feat = tensors["static_feat"].numpy()

    gains_x, gains_y = [], []
    with torch.no_grad():
        for idx in test_idx:
            L = hist_lengths[idx].item()
            hp = hist_padded[idx:idx+1]
            lengths = hist_lengths[idx:idx+1]

            def forward(static_row):
                static = torch.from_numpy(static_row).unsqueeze(0)
                out = model(hp, lengths, static)
                return out[0, PRIMARY_STEP - 1].numpy()

            base_static = static_feat[idx].copy()
            out0 = forward(base_static)

            s_x = base_static.copy(); s_x[-2] += DELTA
            s_y = base_static.copy(); s_y[-1] += DELTA
            out_x = forward(s_x)
            out_y = forward(s_y)
            gains_x.append((out_x[0] - out0[0]) / DELTA)
            gains_y.append((out_y[1] - out0[1]) / DELTA)
    return np.array(gains_x), np.array(gains_y)


def main():
    samples, _ = build_dataset_cached(WEEKS)
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)
    train_val_samples = [s for s in samples if s.game_id in train_val_games]

    examples = build_examples(train_val_samples)
    tensors = build_tensors(examples)
    test_idx = np.where(np.isin(tensors["game_id"], folds[0]))[0]
    anchor_np = tensors["anchor"].numpy()
    base_anchor = np.repeat(anchor_np[:, None, :], tensors["truth"].shape[1], axis=1)
    truth_np = tensors["truth"].numpy()
    mask_np = tensors["mask"].numpy()
    col = PRIMARY_STEP - 1

    sigma_a_raw = estimate_sigma_a(train_val_samples)
    sigma_a = calibrate_sigma_a(train_val_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    Q = build_Q(sigma_a)
    examples_k4 = build_examples_k4(train_val_samples, Q)
    tensors_k4 = build_tensors(examples_k4)
    test_idx_k4 = np.where(np.isin(tensors_k4["game_id"], folds[0]))[0]

    k3 = load_ckpt("K3", 0)
    k4 = load_ckpt("K4", 0)
    k5 = load_ckpt("K5", 0)

    # -------- (a) ball_land gain, K3 vs K4 vs K5
    print("=" * 70)
    print("(a) d(output)/d(ball_land_offset) -- K3 vs K4 vs K5")
    print("=" * 70)
    for label, model, tens, idx in (("K3", k3, tensors, test_idx),
                                    ("K4", k4, tensors_k4, test_idx_k4),
                                    ("K5", k5, tensors, test_idx)):
        gx, gy = ball_land_gain(model, tens, idx)
        print(f"  {label}: d(out_x)/d(bx) mean={gx.mean():.3f} median={np.median(gx):.3f} | "
             f"d(out_y)/d(by) mean={gy.mean():.3f} median={np.median(gy):.3f}")

    # -------- (b) distance to toward_ball_land, clean data, fold 0 (K3 and K5 share tensors)
    print("\n" + "=" * 70)
    print("(b) mean distance to toward_ball_land's prediction, clean data, fold 0")
    print("=" * 70)
    tbl_preds = np.zeros_like(base_anchor[test_idx])
    pos_by_global = {}
    pos = 0
    for sample in train_val_samples:
        n = len(sample.state)
        for j in range(n):
            pos_by_global[pos + j] = (sample, j)
        pos += n
    # build tbl predictions aligned to test_idx order
    sample_cache = {}
    for local_i, global_i in enumerate(test_idx):
        sample, j = pos_by_global[global_i]
        key = id(sample)
        if key not in sample_cache:
            sample_cache[key] = BASELINES["toward_ball_land"](sample, tensors["truth"].shape[1])
        tbl_preds[local_i] = sample_cache[key][j]

    out3 = predict_all(k3, torch.from_numpy(base_anchor), tensors, test_idx)
    pred3 = base_anchor[test_idx] + out3
    out5 = predict_all(k5, torch.from_numpy(base_anchor), tensors, test_idx)
    pred5 = base_anchor[test_idx] + out5

    valid = mask_np[test_idx, col] > 0
    d3 = np.linalg.norm(pred3[valid, col] - tbl_preds[valid, col], axis=-1)
    d5 = np.linalg.norm(pred5[valid, col] - tbl_preds[valid, col], axis=-1)
    print(f"  K3 <-> toward_ball_land: mean distance = {d3.mean():.4f} yd")
    print(f"  K5 <-> toward_ball_land: mean distance = {d5.mean():.4f} yd")
    print(f"  (for reference, K3's own test error @1.9s was ~1.8-2.0, K5's ~2.6-3.0)")

    # -------- (c) ball_land ablation, clean data, fold 0
    print("\n" + "=" * 70)
    print("(c) ball_land ablation (zeroed at inference), clean data, fold 0")
    print("=" * 70)
    static_ablated = tensors["static_feat"].clone()
    static_ablated[:, -2:] = 0.0
    tensors_ablated = dict(tensors)
    tensors_ablated["static_feat"] = static_ablated

    for label, model in (("K3", k3), ("K5", k5)):
        out_normal = predict_all(model, torch.from_numpy(base_anchor), tensors, test_idx)
        pred_normal = base_anchor[test_idx] + out_normal
        err_normal = np.linalg.norm(pred_normal[valid, col] - truth_np[test_idx][valid, col], axis=-1).mean()

        out_ablated = predict_all(model, torch.from_numpy(base_anchor), tensors_ablated, test_idx)
        pred_ablated = base_anchor[test_idx] + out_ablated
        err_ablated = np.linalg.norm(pred_ablated[valid, col] - truth_np[test_idx][valid, col], axis=-1).mean()

        print(f"  {label}: normal={err_normal:.4f}  ablated={err_ablated:.4f}  "
             f"increase={err_ablated - err_normal:.4f} ({100*(err_ablated/err_normal-1):.1f}%)")


if __name__ == "__main__":
    main()
