"""Task M diagnostics: is the sigma=0.2 cliff real noise propagation or
gate saturation / an architectural defect? Forward-pass only, using
Task L's saved checkpoints. No retraining, no test split.

1. Fine sigma grid below 0.2 (K1, K3, all 5 folds, matching Task M's
   own out-of-fold methodology).
2. Pipeline questions answered by direct code citation (no computation).
3. Input out-of-training-range fraction, per history channel, fold 0.
4. Manual GRU gate extraction (update gate z_t), fold 0.
5. ||model_out|| distribution, fold 0, clean vs sigma=0.2.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from config import OUTPUT_DIR, WEEKS
from data_loader import add_kinematics, build_samples, normalise
from degrade import corrupt_input
from model import GRUResidualModel, build_examples
from run_tier1_taskk import build_tensors, compute_base_arrays, masked_error_at_step, predict_all
from run_tier1_taskp import build_predictors_quadrature
from run_tier1_r_convention_check import set_filter_R_convention
from run_tier1_taskm import load_train_val_raw

PRIMARY_STEP = 19
VELOCITY_SOURCE = "derived"
DEGRADE_BASE_SEED_M = 20260929


def load_checkpoints(folds):
    ckpts = {}
    for variant in ("K1", "K3"):
        for i in range(len(folds)):
            ckpt = torch.load(OUTPUT_DIR / "checkpoints" / f"{variant}_fold{i}.pt", weights_only=False)
            model = GRUResidualModel(hidden=ckpt["config"]["hidden"])
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            ckpts[(variant, i)] = model
    return ckpts


def build_at_sigma(sigma, seed, clean_inp, clean_out, sigma_a):
    rng = np.random.default_rng(DEGRADE_BASE_SEED_M + seed)
    degraded_inp = corrupt_input(clean_inp, sigma, rng, VELOCITY_SOURCE)
    degraded_samples = build_samples(degraded_inp, clean_out)
    examples = build_examples(degraded_samples)
    tensors = build_tensors(examples)
    set_filter_R_convention(sigma, VELOCITY_SOURCE, "quadrature")
    predictors = build_predictors_quadrature(sigma_a)
    base_tks = compute_base_arrays(degraded_samples, predictors["tbl_kalman_speed"], examples)
    anchor_np = tensors["anchor"].numpy()
    base_anchor = np.repeat(anchor_np[:, None, :], tensors["truth"].shape[1], axis=1)
    return examples, tensors, {"K1": base_tks, "K3": base_anchor}


def main():
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)

    clean_inp, clean_out = load_train_val_raw(train_val_games)
    clean_samples_for_sigma = build_samples(clean_inp, clean_out)
    from kalman import calibrate_sigma_a, estimate_sigma_a
    sigma_a_raw = estimate_sigma_a(clean_samples_for_sigma)
    sigma_a = calibrate_sigma_a(clean_samples_for_sigma, sigma_a_hi=sigma_a_raw, target=2.0)

    checkpoints = load_checkpoints(folds)

    # ================================================================
    # 1. fine sigma grid below 0.2
    # ================================================================
    print("=" * 70)
    print("1. Fine sigma grid, K1 and K3, all 5 folds out-of-fold, @1.9s")
    print("=" * 70)
    fine_sigmas = [0.0, 0.02, 0.05, 0.10, 0.15, 0.2]
    for sigma in fine_sigmas:
        examples, tensors, bases_np = build_at_sigma(sigma, 0, clean_inp, clean_out, sigma_a)
        truth_np = tensors["truth"].numpy()
        mask_np = tensors["mask"].numpy()
        game_id_arr = tensors["game_id"]
        n = len(examples)
        oof_pred = {v: np.full((n, truth_np.shape[1], 2), np.nan, dtype=np.float32) for v in ("K1", "K3")}
        for i, fold_games in enumerate(folds):
            test_idx = np.where(np.isin(game_id_arr, fold_games))[0]
            for variant in ("K1", "K3"):
                model = checkpoints[(variant, i)]
                base_t = torch.from_numpy(bases_np[variant])
                model_out = predict_all(model, base_t, tensors, test_idx)
                oof_pred[variant][test_idx] = bases_np[variant][test_idx] + model_out
        line = f"sigma={sigma:5.2f}  "
        for variant in ("K1", "K3"):
            err, n_v = masked_error_at_step(oof_pred[variant], truth_np, mask_np, PRIMARY_STEP)
            line += f" {variant}={err:8.4f} (n={n_v})"
        print(line)

    # ================================================================
    # 2. pipeline questions -- code citation only
    # ================================================================
    print("\n" + "=" * 70)
    print("2. Pipeline questions -- from code")
    print("=" * 70)
    print("""
(a) Noise applied before or after the anchor-relative transform?
    BEFORE. run_tier1_taskm.main(): corrupt_input(clean_inp, sigma, rng,
    "derived") runs on the raw absolute-coordinate input DataFrame, then
    build_samples(degraded_inp, clean_out) builds PlaySamples from it,
    then build_examples (model.py) computes rel_x = hist_x - anchor_x
    where BOTH hist_x and anchor_x come from that same already-corrupted
    data (anchor is literally history's own last row). Consequence:
    rel_x[t] = (x[t]+eps[t]) - (x[T]+eps[T]) = (x[t]-x[T]) + (eps[t]-eps[T]).
    Every relative-position feature in a track shares the SAME anchor
    noise term eps[T], subtracted with the same sign at every timestep --
    a constant per-track shift, not fresh independent per-frame noise.
    Velocity, by contrast, differences CONSECUTIVE frames
    (v[t] = (x[t]+eps[t] - x[t-1]-eps[t-1])/DT), so its noise is
    independent per step and amplified by sqrt(2)/DT ~= 14.1x, per
    degrade.py's own docstring. Velocity is the dominant amplification
    path, not position.

(b) Are input features standardised (z-scored)?
    NO. model.py's GRUResidualModel has no normalisation layer anywhere
    -- feeds raw yards/(yd/s) values directly: nn.GRU(HIST_DIM, hidden,
    ...) followed by nn.Sequential(Linear, ReLU, Linear). No BatchNorm,
    LayerNorm, or manual mean/std scaling at any point in model.py or
    run_tier1_taskk.py's tensor-building. Out-of-distribution magnitude
    inputs reach the GRU's weight matrices completely unscaled.

(c) Is the model's output clipped or bounded anywhere?
    NO. The decoder's final layer is `nn.Linear(mlp_hidden, 2)` with no
    activation after it -- no tanh, no clamp, no sigmoid-scaled range.
    model_out is unbounded in principle; its actual magnitude is
    whatever the learned weights produce for the given input.
""")

    # ================================================================
    # 3/4/5 -- fold 0 only, clean vs sigma=0.2
    # ================================================================
    print("=" * 70)
    print("3/4/5. Input distribution, gate saturation, output magnitude -- fold 0, K1 and K3")
    print("=" * 70)

    examples_clean, tensors_clean, bases_clean = build_at_sigma(0.0, 0, clean_inp, clean_out, sigma_a)
    examples_02, tensors_02, bases_02 = build_at_sigma(0.2, 0, clean_inp, clean_out, sigma_a)

    game_id_arr_clean = tensors_clean["game_id"]
    game_id_arr_02 = tensors_02["game_id"]
    fold0_games = folds[0]
    # everything not in fold0's test games is train+val for fold 0's model;
    # we want the TRAIN portion specifically for "range seen in training"
    val_games = set(folds[1])
    test_games = set(fold0_games)
    train_games = train_val_games - test_games - val_games

    train_idx_clean = np.where(np.isin(game_id_arr_clean, list(train_games)))[0]
    test_idx_clean = np.where(np.isin(game_id_arr_clean, fold0_games))[0]
    test_idx_02 = np.where(np.isin(game_id_arr_02, fold0_games))[0]

    hist_padded_clean = tensors_clean["hist_padded"].numpy()
    hist_lengths_clean = tensors_clean["hist_lengths"].numpy()
    hist_padded_02 = tensors_02["hist_padded"].numpy()
    hist_lengths_02 = tensors_02["hist_lengths"].numpy()

    channel_names = ["rel_x", "rel_y", "vx", "vy"]

    # -------- 3. out-of-training-range fraction
    print("\n--- 3. Input out-of-training-range fraction, per channel ---")
    train_min = np.full(4, np.inf)
    train_max = np.full(4, -np.inf)
    for idx in train_idx_clean:
        L = hist_lengths_clean[idx]
        vals = hist_padded_clean[idx, :L, :]
        train_min = np.minimum(train_min, vals.min(axis=0))
        train_max = np.maximum(train_max, vals.max(axis=0))
    print(f"training range (fold 0's train set, clean), per channel:")
    for c, name in enumerate(channel_names):
        print(f"  {name}: [{train_min[c]:.3f}, {train_max[c]:.3f}]")

    for label, hist_padded, hist_lengths, test_idx in (
        ("sigma=0.0", hist_padded_clean, hist_lengths_clean, test_idx_clean),
        ("sigma=0.2", hist_padded_02, hist_lengths_02, test_idx_02),
    ):
        print(f"\n  {label} (fold-0 test set):")
        for c, name in enumerate(channel_names):
            vals = []
            for idx in test_idx:
                L = hist_lengths[idx]
                vals.append(hist_padded[idx, :L, c])
            vals = np.concatenate(vals)
            frac_out = ((vals < train_min[c]) | (vals > train_max[c])).mean()
            print(f"    {name}: {frac_out:.4f} fraction outside training range "
                 f"(value range seen: [{vals.min():.2f}, {vals.max():.2f}])")

    # -------- 4. gate saturation (manual GRU forward)
    print("\n--- 4. Update-gate (z_t) saturation, manual GRU forward ---")

    def gru_gate_stats(model, hist_padded, hist_lengths, idx, hidden):
        w_ih = model.gru.weight_ih_l0.detach().numpy()   # (3*hidden, HIST_DIM)
        w_hh = model.gru.weight_hh_l0.detach().numpy()   # (3*hidden, hidden)
        b_ih = model.gru.bias_ih_l0.detach().numpy()
        b_hh = model.gru.bias_hh_l0.detach().numpy()
        W_ir, W_iz, W_in = w_ih[:hidden], w_ih[hidden:2*hidden], w_ih[2*hidden:]
        W_hr, W_hz, W_hn = w_hh[:hidden], w_hh[hidden:2*hidden], w_hh[2*hidden:]
        b_ir, b_iz, b_in = b_ih[:hidden], b_ih[hidden:2*hidden], b_ih[2*hidden:]
        b_hr, b_hz, b_hn = b_hh[:hidden], b_hh[hidden:2*hidden], b_hh[2*hidden:]

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        all_z = []
        for i in idx:
            L = hist_lengths[i]
            seq = hist_padded[i, :L, :]     # (L, HIST_DIM)
            h = np.zeros(hidden, dtype=np.float64)
            for t in range(L):
                x_t = seq[t].astype(np.float64)
                r = sigmoid(W_ir @ x_t + b_ir + W_hr @ h + b_hr)
                z = sigmoid(W_iz @ x_t + b_iz + W_hz @ h + b_hz)
                n = np.tanh(W_in @ x_t + b_in + r * (W_hn @ h + b_hn))
                h = (1 - z) * n + z * h
                all_z.append(z)
        return np.concatenate(all_z)

    rng_sub = np.random.default_rng(0)
    sample_idx_clean = rng_sub.choice(test_idx_clean, size=min(100, len(test_idx_clean)), replace=False)
    sample_idx_02 = sample_idx_clean   # same tracks, different sigma's degraded tensor

    for variant in ("K1", "K3"):
        model = checkpoints[(variant, 0)]
        hidden = model.hidden
        z_clean = gru_gate_stats(model, hist_padded_clean, hist_lengths_clean, sample_idx_clean, hidden)
        z_02 = gru_gate_stats(model, hist_padded_02, hist_lengths_02, sample_idx_02, hidden)
        for label, z in (("clean", z_clean), ("sigma=0.2", z_02)):
            sat = ((z < 0.01) | (z > 0.99)).mean()
            print(f"  {variant} {label}: mean z={z.mean():.4f}  "
                 f"fraction saturated (<0.01 or >0.99)={sat:.4f}")

    # -------- 5. output magnitude
    print("\n--- 5. ||model_out|| distribution, fold 0, @1.9s ---")
    col = PRIMARY_STEP - 1
    for variant in ("K1", "K3"):
        model = checkpoints[(variant, 0)]
        mag_stats = {}
        for label, tensors, bases_np, test_idx in (
            ("clean", tensors_clean, bases_clean, test_idx_clean),
            ("sigma=0.2", tensors_02, bases_02, test_idx_02),
        ):
            base_t = torch.from_numpy(bases_np[variant])
            model_out = predict_all(model, base_t, tensors, test_idx)
            mag = np.linalg.norm(model_out[:, col], axis=-1)
            print(f"  {variant} {label}: mean={mag.mean():.4f} median={np.median(mag):.4f} "
                 f"p99={np.percentile(mag,99):.4f} max={mag.max():.4f}")


if __name__ == "__main__":
    main()
