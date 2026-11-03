"""Tier 2A: drift-detection signals, test split, forward-pass only.

Four candidate signals computed with NO ground truth, across the
already-published test-split degradation sweep (same sigma grid as
run_tier1_test_degrade.py). Every threshold comes from training data
only, or from theory (signal 4) -- never from test. See the pre-flight
notes in the chat transcript for the corrections made to the original
spec (H-matrix in the NIS formula, frozen R for signal 4, the feature
list for signal 1, the "3-baseline" caveat for signal 2's hard lead).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from baselines import BASELINES
from config import DT, OUTPUT_DIR, WEEKS
from data_loader import add_kinematics, build_samples, load_raw, normalise
from degrade import corrupt_input
from kalman import R_POS as KALMAN_R_POS_MATRIX
from kalman import H_POS, _run_filter, build_Q, calibrate_sigma_a, estimate_sigma_a
from model import GRUResidualModel, build_examples
from run_tier1_taskk import build_tensors, predict_all
from run_tier1_taskm import load_train_val_raw
from run_tier1_taskp import build_predictors_quadrature
from run_tier1_r_convention_check import set_filter_R_convention

PRIMARY_STEP = 19
COL = PRIMARY_STEP - 1
SIGMA_GRID = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 5.0]
SEEDS = [0, 1, 2, 3, 4]
DEGRADE_BASE_SEED_M = 20260929
FEATURE_NAMES = ["rel_x", "rel_y", "vx", "vy"]
PSI_ALARM = 0.2
SPC_K = 3.0


# ================================================================ PSI
def psi(actual: np.ndarray, expected: np.ndarray, bin_edges: np.ndarray) -> float:
    eps = 1e-6
    a_counts, _ = np.histogram(actual, bins=bin_edges)
    e_counts, _ = np.histogram(expected, bins=bin_edges)
    a_frac = a_counts / max(len(actual), 1) + eps
    e_frac = e_counts / max(len(expected), 1) + eps
    return float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))


def decile_edges(ref: np.ndarray) -> np.ndarray:
    qs = np.linspace(0, 1, 11)
    edges = np.quantile(ref, qs)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    return np.unique(edges) if len(np.unique(edges)) > 2 else np.array([edges[0], edges[0] + 1e-6, edges[-1]])


# ================================================================ loaders
def load_final(variant):
    ckpt = torch.load(OUTPUT_DIR / "checkpoints" / f"{variant}_FINAL_fold0.pt", weights_only=False)
    model = GRUResidualModel(hidden=ckpt["config"]["hidden"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_train_games():
    with open(OUTPUT_DIR / "final_val_games.json") as f:
        fv = json.load(f)
    return fv["final_train_games"]


def load_raw_games(games: set):
    inp, out = load_raw(weeks=WEEKS)
    inp = inp[inp["game_id"].isin(games)].reset_index(drop=True)
    out = out[out["game_id"].isin(games)].reset_index(drop=True)
    inp, out = normalise(inp, out)
    inp = add_kinematics(inp)
    return inp, out


# ================================================================ main
def main():
    with open(OUTPUT_DIR / "splits.json") as f:
        splits = json.load(f)
    test_games = set(splits["test"])
    train_val_games = set(splits["train"] + splits["val"])
    final_train_games = set(int(g) for g in load_train_games())

    print("loading clean training population (final-model train games) for baselines...")
    train_inp, train_out = load_raw_games(final_train_games)
    train_samples = build_samples(train_inp, train_out)

    print("loading clean test population (sigma=0) for the benign-shift construction...")
    clean_test_inp, clean_test_out = load_raw_games(test_games)

    print("estimating frozen sigma_a from train+val...")
    tv_inp, tv_out = load_raw_games(train_val_games)
    tv_samples = build_samples(tv_inp, tv_out)
    sigma_a_raw = estimate_sigma_a(tv_samples)
    sigma_a = calibrate_sigma_a(tv_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    Q = build_Q(sigma_a)
    print(f"sigma_a = {sigma_a:.4f}")

    k3 = load_final("K3")
    k5 = load_final("K5")

    # ------------------------------------------------ training baselines
    print("\nbuilding training-population baselines (clean, final-model train games)...")
    train_examples = build_examples(train_samples)
    train_tensors = build_tensors(train_examples)
    train_idx_all = np.arange(len(train_examples))
    anchor_tr = train_tensors["anchor"].numpy()
    base_anchor_tr = np.repeat(anchor_tr[:, None, :], train_tensors["truth"].shape[1], axis=1)
    base_t_tr = torch.from_numpy(base_anchor_tr)

    # signal-1 reference bins (rel_x, rel_y, vx, vy), pooled over valid (track, step)
    hist_padded_tr = train_tensors["hist_padded"].numpy()
    hist_lengths_tr = train_tensors["hist_lengths"].numpy()
    ref_feature_pool = {name: [] for name in FEATURE_NAMES}
    for i in range(len(train_examples)):
        L = hist_lengths_tr[i]
        for c, name in enumerate(FEATURE_NAMES):
            ref_feature_pool[name].append(hist_padded_tr[i, :L, c])
    ref_feature_pool = {k: np.concatenate(v) for k, v in ref_feature_pool.items()}
    ref_edges = {k: decile_edges(v) for k, v in ref_feature_pool.items()}
    print("signal 1 reference pools (n):", {k: len(v) for k, v in ref_feature_pool.items()})

    # signal-2 training baseline: |model_pred - tbl_kalman_speed_pred| at 1.9s, clean
    set_filter_R_convention(0.0, "derived", "quadrature")
    predictors_clean = build_predictors_quadrature(sigma_a)
    base_tks_tr = np.zeros_like(base_anchor_tr)
    pos = 0
    for sample in train_samples:
        n = len(sample.state)
        base_tks_tr[pos:pos+n] = predictors_clean["tbl_kalman_speed"](sample, train_tensors["truth"].shape[1])
        pos += n

    sig2_baseline = {}
    sig3_ref_pool = {}
    model_out_tr_cache = {}
    for name, model in (("K3", k3), ("K5", k5)):
        model_out = predict_all(model, base_t_tr, train_tensors, train_idx_all)
        model_out_tr_cache[name] = model_out
        pred = base_anchor_tr[train_idx_all] + model_out
        dist = np.linalg.norm(pred[:, COL] - base_tks_tr[:, COL], axis=-1)
        sig2_baseline[name] = (float(dist.mean()), float(dist.std()))
        disp_mag = np.linalg.norm(model_out[:, COL], axis=-1)
        sig3_ref_pool[name] = disp_mag
    print("signal 2 training baseline (mean, sd):", sig2_baseline)

    # signal-4 theory threshold: frozen R, dof=2 (position-only H), normal
    # approximation to the batch-mean of N pooled NIS values
    set_filter_R_convention(0.0, "derived", "quadrature")   # frozen: sigma_NGS only, never updated per true sigma
    dof = H_POS.shape[0]
    print(f"\nsignal 4: frozen R (sigma_NGS only), dof={dof}, theoretical E[NIS]={dof}")

    # ------------------------------------------------ sweep
    rows = []
    for sigma in SIGMA_GRID:
        seeds = SEEDS if sigma > 0 else [0]
        sig1_vals, sig2_vals, sig3_vals, sig4_vals = [], {}, {}, []
        sig2_vals = {"K3": [], "K5": []}
        sig3_vals = {"K3": [], "K5": []}
        feature_pool = {name: [] for name in FEATURE_NAMES}

        for seed in seeds:
            rng = np.random.default_rng(DEGRADE_BASE_SEED_M + seed)
            degraded_inp = corrupt_input(clean_test_inp, sigma, rng, "derived")
            degraded_samples = build_samples(degraded_inp, clean_test_out)

            examples = build_examples(degraded_samples)
            tensors = build_tensors(examples)
            anchor = tensors["anchor"].numpy()
            base_anchor = np.repeat(anchor[:, None, :], tensors["truth"].shape[1], axis=1)
            base_t = torch.from_numpy(base_anchor)
            idx_all = np.arange(len(examples))

            # signal 1: pool rel_x, rel_y, vx, vy
            hist_padded = tensors["hist_padded"].numpy()
            hist_lengths = tensors["hist_lengths"].numpy()
            for i in range(len(examples)):
                L = hist_lengths[i]
                for c, name in enumerate(FEATURE_NAMES):
                    feature_pool[name].append(hist_padded[i, :L, c])

            # adaptive R for the actual degraded baseline (tbl_kalman_speed as a
            # predictor, not as the monitor -- this IS the real deployed base)
            set_filter_R_convention(sigma, "derived", "quadrature")
            predictors = build_predictors_quadrature(sigma_a)
            base_tks = np.zeros_like(base_anchor)
            pos = 0
            for sample in degraded_samples:
                n = len(sample.state)
                base_tks[pos:pos+n] = predictors["tbl_kalman_speed"](sample, tensors["truth"].shape[1])
                pos += n

            for name, model in (("K3", k3), ("K5", k5)):
                model_out = predict_all(model, base_t, tensors, idx_all)
                pred = base_anchor[idx_all] + model_out
                dist = np.linalg.norm(pred[:, COL] - base_tks[:, COL], axis=-1)
                sig2_vals[name].append(dist)
                disp_mag = np.linalg.norm(model_out[:, COL], axis=-1)
                sig3_vals[name].append(disp_mag)

            # signal 4: frozen R, run the position-only filter, collect NIS
            set_filter_R_convention(0.0, "derived", "quadrature")   # frozen
            nis_sink = []
            for sample in degraded_samples:
                hist_by_player = dict(tuple(sample.history.groupby("player_id", sort=False)))
                for player_id in sample.state["player_id"]:
                    grp = hist_by_player.get(player_id)
                    if grp is None or len(grp) < 3:
                        continue
                    _run_filter(grp, Q, use_velocity=False, nis_sink=nis_sink)
            sig4_vals.extend(nis_sink)

        feature_pool = {k: np.concatenate(v) for k, v in feature_pool.items()}
        psi_per_feature = {name: psi(feature_pool[name], ref_feature_pool[name], ref_edges[name])
                           for name in FEATURE_NAMES}
        psi_max = max(psi_per_feature.values())

        row = {"sigma": sigma, "psi_max": psi_max, **{f"psi_{k}": v for k, v in psi_per_feature.items()},
              "nis_mean": float(np.mean(sig4_vals)), "nis_n": len(sig4_vals)}
        for name in ("K3", "K5"):
            d = np.concatenate(sig2_vals[name])
            m = np.concatenate(sig3_vals[name])
            row[f"sig2_{name}_mean"] = float(d.mean())
            row[f"sig3_{name}_psi"] = psi(m, sig3_ref_pool[name], decile_edges(sig3_ref_pool[name]))
        rows.append(row)
        print(f"sigma={sigma}: psi_max={psi_max:.4f} nis_mean={row['nis_mean']:.4f} "
             f"sig2_K3={row['sig2_K3_mean']:.3f} sig2_K5={row['sig2_K5_mean']:.3f} "
             f"sig3_K3_psi={row['sig3_K3_psi']:.4f} sig3_K5_psi={row['sig3_K5_psi']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "tier2a_signals_raw.csv", index=False)
    print("\nFull signal table:")
    print(df.round(4).to_string(index=False))

    # save objects needed by the follow-up (thresholds, benign-shift check)
    import pickle
    with open(OUTPUT_DIR / "tier2a_state.pkl", "wb") as f:
        pickle.dump({
            "sig2_baseline": sig2_baseline,
            "sig3_ref_pool": sig3_ref_pool,
            "ref_feature_pool": ref_feature_pool,
            "ref_edges": ref_edges,
            "dof": dof,
            "sigma_a": sigma_a,
        }, f)
    print("\nsaved outputs/tier2a_signals_raw.csv and outputs/tier2a_state.pkl")


if __name__ == "__main__":
    main()
