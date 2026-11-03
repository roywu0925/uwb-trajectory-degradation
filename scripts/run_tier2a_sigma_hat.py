"""Tier 2A, item 3: rebuild the Kalman-innovation signal as a calibrated
sigma_hat estimator instead of a theoretical alarm. Steps 3a-3c: training-
only calibration curve, inversion applied to the already-run test sweep
(recomputed at per-play granularity, since the first pass only saved the
aggregate), and batch-size precision.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, WEEKS
from data_loader import add_kinematics, build_samples, load_raw, normalise
from degrade import corrupt_input
from kalman import H_POS, _run_filter, build_Q, calibrate_sigma_a, estimate_sigma_a
from run_tier1_r_convention_check import set_filter_R_convention
from run_tier1_taskm import load_train_val_raw

CALIBRATION_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
TEST_GRID = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 5.0]
SEEDS = [0, 1, 2, 3, 4]
DEGRADE_BASE_SEED_M = 20260929
BATCH_SIZES = [1, 10, 50, 200]


def nis_by_play(samples, Q):
    """dict play_key -> list of NIS values (all tracks' update-steps
    pooled for that play), frozen R (whatever set_filter_R_convention
    last set, caller's responsibility)."""
    out = {}
    for sample in samples:
        key = (sample.game_id, sample.play_id)
        vals = []
        hist_by_player = dict(tuple(sample.history.groupby("player_id", sort=False)))
        for player_id in sample.state["player_id"]:
            grp = hist_by_player.get(player_id)
            if grp is None or len(grp) < 3:
                continue
            _run_filter(grp, Q, use_velocity=False, nis_sink=vals)
        if vals:
            out[key] = vals
    return out


def load_train_games():
    with open(OUTPUT_DIR / "final_val_games.json") as f:
        fv = json.load(f)
    return set(int(g) for g in fv["final_train_games"])


def load_raw_games(games: set):
    inp, out = load_raw(weeks=WEEKS)
    inp = inp[inp["game_id"].isin(games)].reset_index(drop=True)
    out = out[out["game_id"].isin(games)].reset_index(drop=True)
    inp, out = normalise(inp, out)
    inp = add_kinematics(inp)
    return inp, out


def main():
    with open(OUTPUT_DIR / "splits.json") as f:
        splits = json.load(f)
    test_games = set(splits["test"])
    train_val_games = set(splits["train"] + splits["val"])
    final_train_games = load_train_games()

    print("loading train+val for frozen sigma_a...")
    tv_inp, tv_out = load_raw_games(train_val_games)
    tv_samples = build_samples(tv_inp, tv_out)
    sigma_a_raw = estimate_sigma_a(tv_samples)
    sigma_a = calibrate_sigma_a(tv_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    Q = build_Q(sigma_a)
    print(f"sigma_a = {sigma_a:.4f}")

    print("\nloading clean training population (final-model train games)...")
    train_inp, train_out = load_raw_games(final_train_games)

    # ================================================================ 3a
    print("\n" + "=" * 70)
    print("3a. Calibration curve, TRAINING data only, frozen R")
    print("=" * 70)
    set_filter_R_convention(0.0, "derived", "quadrature")   # frozen, set once, never touched again

    calib_rows = []
    calib_by_play_master = {}   # sigma -> {play_key: [nis,...]}
    for sigma in CALIBRATION_GRID:
        pooled = []
        by_play_all_seeds = {}
        seeds = SEEDS if sigma > 0 else [0]
        for seed in seeds:
            rng = np.random.default_rng(DEGRADE_BASE_SEED_M + seed)
            degraded_inp = corrupt_input(train_inp, sigma, rng, "derived")
            degraded_samples = build_samples(degraded_inp, train_out)
            by_play = nis_by_play(degraded_samples, Q)
            for k, v in by_play.items():
                pooled.extend(v)
                by_play_all_seeds.setdefault((seed, k), v)
        calib_by_play_master[sigma] = by_play_all_seeds
        pooled = np.array(pooled)
        calib_rows.append({"sigma": sigma, "nis_mean": pooled.mean(), "nis_std": pooled.std(), "n": len(pooled)})
        print(f"  sigma={sigma}: nis_mean={pooled.mean():.4f} nis_std={pooled.std():.4f} n={len(pooled)}")

    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv(OUTPUT_DIR / "tier2a_calibration_curve.csv", index=False)

    diffs = np.diff(calib_df["nis_mean"].to_numpy())
    monotone = bool(np.all(diffs > 0))
    print(f"\nmonotone increasing? {monotone}")
    if not monotone:
        bad = np.where(diffs <= 0)[0]
        for i in bad:
            print(f"  NOT monotone between sigma={calib_df['sigma'].iloc[i]} "
                 f"(nis={calib_df['nis_mean'].iloc[i]:.4f}) and "
                 f"sigma={calib_df['sigma'].iloc[i+1]} (nis={calib_df['nis_mean'].iloc[i+1]:.4f})")

    # ================================================================ 3b
    print("\n" + "=" * 70)
    print("3b. Invert on the (re-derived, per-play) test sweep -- in range only (sigma<=2)")
    print("=" * 70)
    calib_sigma = calib_df["sigma"].to_numpy()
    calib_nis = calib_df["nis_mean"].to_numpy()

    def invert(nis_value):
        # calib_nis is monotone increasing in calib_sigma (checked above) --
        # np.interp needs x increasing, which calib_nis is by construction
        return float(np.interp(nis_value, calib_nis, calib_sigma))

    print("loading clean test population for the per-play re-derivation...")
    clean_test_inp, clean_test_out = load_raw_games(test_games)

    test_by_play_master = {}
    bias_rows = []
    for sigma in TEST_GRID:
        pooled = []
        by_play_all_seeds = {}
        seeds = SEEDS if sigma > 0 else [0]
        for seed in seeds:
            rng = np.random.default_rng(DEGRADE_BASE_SEED_M + seed)
            degraded_inp = corrupt_input(clean_test_inp, sigma, rng, "derived")
            degraded_samples = build_samples(degraded_inp, clean_test_out)
            by_play = nis_by_play(degraded_samples, Q)
            for k, v in by_play.items():
                pooled.extend(v)
                by_play_all_seeds[(seed, k)] = v
        test_by_play_master[sigma] = by_play_all_seeds
        pooled = np.array(pooled)
        nis_mean = pooled.mean()
        sigma_hat = invert(nis_mean)
        in_range = sigma <= 2.0
        bias_rows.append({"true_sigma": sigma, "nis_mean": nis_mean, "sigma_hat": sigma_hat,
                          "bias": sigma_hat - sigma, "in_calibration_range": in_range, "n": len(pooled)})
        flag = "" if in_range else "  <-- EXTRAPOLATION, outside 0-2 calibration range"
        print(f"  true_sigma={sigma}: nis_mean={nis_mean:.4f} sigma_hat={sigma_hat:.4f} "
             f"bias={sigma_hat-sigma:+.4f}{flag}")

    bias_df = pd.DataFrame(bias_rows)
    bias_df.to_csv(OUTPUT_DIR / "tier2a_sigma_hat_bias.csv", index=False)

    # ================================================================ 3c
    print("\n" + "=" * 70)
    print("3c. sigma_hat precision by batch size (plays), from TRAINING calibration data")
    print("=" * 70)
    precision_rows = []
    for sigma in CALIBRATION_GRID:
        by_play = calib_by_play_master[sigma]
        play_keys = list(by_play.keys())
        n_plays = len(play_keys)
        for batch_size in BATCH_SIZES:
            if batch_size > n_plays:
                continue
            rng = np.random.default_rng(0)
            sigma_hats = []
            for _ in range(500):
                chosen = rng.choice(n_plays, size=batch_size, replace=False)
                vals = np.concatenate([by_play[play_keys[i]] for i in chosen])
                sigma_hats.append(invert(vals.mean()))
            sigma_hats = np.array(sigma_hats)
            precision_rows.append({"true_sigma": sigma, "batch_size_plays": batch_size,
                                   "sigma_hat_mean": sigma_hats.mean(), "sigma_hat_std": sigma_hats.std(),
                                   "n_plays_available": n_plays})
    precision_df = pd.DataFrame(precision_rows)
    precision_df.to_csv(OUTPUT_DIR / "tier2a_batch_precision.csv", index=False)
    print(precision_df.round(4).to_string(index=False))

    # persist state for 3d/3e
    with open(OUTPUT_DIR / "tier2a_sigma_hat_state.pkl", "wb") as f:
        pickle.dump({
            "calib_sigma": calib_sigma, "calib_nis": calib_nis,
            "sigma_a": sigma_a, "Q": Q,
        }, f)
    print("\nsaved calibration curve, bias table, batch-precision table, and inversion state.")


if __name__ == "__main__":
    main()
