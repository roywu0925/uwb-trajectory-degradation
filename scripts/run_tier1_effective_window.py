"""Effective baseline length / smoothing-window test (forward-pass only,
no retraining): does K5's suppressed velocity gain mean it derived its
own internal filter from the position history, the way a GRU's update
h = (1-z)h_{t-1} + z*h~ can act as an exponentially weighted average?

Two position-history perturbations, K3/K4/K5:
  A. single frame (the middle history frame) shifted by delta
  B. coherent ramp across the WHOLE history, slope delta per frame,
     anchored to zero at the anchor frame (preserving the anchor
     invariant), i.e. the same per-step magnitude as A but sustained
     through the entire window instead of applied once.

B/A near 1 => short effective window, no internal smoothing.
B/A >> 1   => long effective window: single-frame noise suppressed,
             coherent trend passed through -- the Kalman-filter
             signature.
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
DELTA = 0.1   # yards
WINDOW = 10   # fixed-length window for both A and B, same for every track/variant


def perturb_single_frame(seq: np.ndarray, L: int, axis: int, delta: float) -> np.ndarray:
    """A: one frame, in the middle of the fixed last-WINDOW frames before
    the anchor -- same local region B operates on, isolating "one blip"
    vs "sustained ramp" without track length as a confound."""
    out = seq.copy()
    k = L - 1 - WINDOW // 2
    out[k, axis] += delta
    return out


def perturb_ramp(seq: np.ndarray, L: int, axis: int, delta: float) -> np.ndarray:
    """B: fixed-length (WINDOW frames) ramp, zero at the anchor, slope
    `delta` per step -- same per-step magnitude as A, same total
    amplitude (delta * (WINDOW-1)) regardless of track length. This is
    the fix for the original design, whose ramp amplitude scaled with
    track length and confounded the A-vs-B comparison."""
    out = seq.copy()
    start = L - WINDOW
    t = np.arange(start, L)
    ramp = delta * (t - (L - 1))   # zero at the anchor, slope `delta` per step
    out[start:L, axis] += ramp
    return out


def measure(model, hist_padded, hist_lengths, static_feat, test_idx, axis, perturb_fn):
    out_mags = []
    with torch.no_grad():
        for idx in test_idx:
            L = hist_lengths[idx].item()
            if L < 3:
                continue
            base_seq = hist_padded[idx, :L, :].numpy().copy()

            def forward(seq):
                hp = hist_padded[idx:idx+1].clone()
                hp[0, :L, :] = torch.from_numpy(seq)
                lengths = hist_lengths[idx:idx+1]
                static = static_feat[idx:idx+1]
                out = model(hp, lengths, static)
                return out[0, PRIMARY_STEP - 1].numpy()

            out0 = forward(base_seq)
            seq_p = perturb_fn(base_seq, L, axis, DELTA)
            out_p = forward(seq_p)
            out_mags.append(np.linalg.norm(out_p - out0))
    return np.array(out_mags)


def load_ckpt(variant, fold_id=0):
    ckpt = torch.load(OUTPUT_DIR / "checkpoints" / f"{variant}_fold{fold_id}.pt", weights_only=False)
    model = GRUResidualModel(hidden=ckpt["config"]["hidden"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    samples, _ = build_dataset_cached(WEEKS)
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)
    train_val_samples = [s for s in samples if s.game_id in train_val_games]

    examples_std = build_examples(train_val_samples)
    tensors_std = build_tensors(examples_std)
    test_idx_std = np.where(np.isin(tensors_std["game_id"], folds[0]))[0]
    # restrict to longer tracks so the "middle frame" and full-ramp are meaningful
    hist_lengths_std = tensors_std["hist_lengths"].numpy()
    test_idx_std = test_idx_std[hist_lengths_std[test_idx_std] >= 10]

    sigma_a_raw = estimate_sigma_a(train_val_samples)
    sigma_a = calibrate_sigma_a(train_val_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    Q = build_Q(sigma_a)
    examples_k4 = build_examples_k4(train_val_samples, Q)
    tensors_k4 = build_tensors(examples_k4)
    test_idx_k4 = np.where(np.isin(tensors_k4["game_id"], folds[0]))[0]
    hist_lengths_k4 = tensors_k4["hist_lengths"].numpy()
    test_idx_k4 = test_idx_k4[hist_lengths_k4[test_idx_k4] >= 10]

    print(f"{'variant':6s} {'axis':6s} {'A mean':>10s} {'A std':>8s} "
         f"{'B mean':>10s} {'B std':>8s} {'B/A':>8s}")
    for variant, examples, tensors, test_idx in (
        ("K3", examples_std, tensors_std, test_idx_std),
        ("K4", examples_k4, tensors_k4, test_idx_k4),
        ("K5", examples_std, tensors_std, test_idx_std),
    ):
        model = load_ckpt(variant, 0)
        hist_padded = tensors["hist_padded"]
        hist_lengths = tensors["hist_lengths"]
        static_feat = tensors["static_feat"]

        for axis, axis_name in ((0, "rel_x"), (1, "rel_y")):
            a_mags = measure(model, hist_padded, hist_lengths, static_feat, test_idx, axis, perturb_single_frame)
            b_mags = measure(model, hist_padded, hist_lengths, static_feat, test_idx, axis, perturb_ramp)
            ratio = b_mags.mean() / a_mags.mean()
            print(f"{variant:6s} {axis_name:6s} {a_mags.mean():10.4f} {a_mags.std():8.4f} "
                 f"{b_mags.mean():10.4f} {b_mags.std():8.4f} {ratio:8.3f}")


if __name__ == "__main__":
    main()
