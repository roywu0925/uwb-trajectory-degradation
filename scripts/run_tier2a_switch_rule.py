"""Tier 2A items 3d/3e: the Q confound measured in yards, and the
end-to-end switch rule (sigma_hat vs. the CV-derived crossing, 1.0552 --
never the test crossing, to avoid circularity) evaluated against an
oracle that knows the true test crossing (0.9338).
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, WEEKS
from data_loader import add_kinematics, build_samples, load_raw, normalise
from degrade import corrupt_input
from kalman import _run_filter
from model import build_examples
from run_tier1_r_convention_check import set_filter_R_convention
from run_tier2a_sigma_hat import DEGRADE_BASE_SEED_M, SEEDS, TEST_GRID, nis_by_play

CV_CROSSING = 1.0552
TEST_CROSSING = 0.9338


def invert(nis_value, calib_nis, calib_sigma):
    return float(np.interp(nis_value, calib_nis, calib_sigma))


def main():
    with open(OUTPUT_DIR / "tier2a_sigma_hat_state.pkl", "rb") as f:
        state = pickle.load(f)
    calib_sigma, calib_nis, Q = state["calib_sigma"], state["calib_nis"], state["Q"]

    with open(OUTPUT_DIR / "splits.json") as f:
        splits = json.load(f)
    test_games = set(splits["test"])

    # ================================================================ 3d
    print("=" * 70)
    print("3d. Q confound: role-stratified speed resample, true sigma = 0")
    print("=" * 70)
    inp, out = load_raw(weeks=WEEKS)
    inp = inp[inp["game_id"].isin(test_games)].reset_index(drop=True)
    out = out[out["game_id"].isin(test_games)].reset_index(drop=True)
    inp, out = normalise(inp, out)
    inp = add_kinematics(inp)
    samples0 = build_samples(inp, out)
    examples0 = build_examples(samples0)
    roles = np.array([e.role for e in examples0])
    n0 = len(examples0)

    hist_lengths0 = np.array([e.hist_feat.shape[0] for e in examples0])
    anchor_speed = np.array([np.hypot(e.hist_feat[-1, 2], e.hist_feat[-1, 3]) for e in examples0])

    weights = np.zeros(n0)
    for r in np.unique(roles):
        m = roles == r
        w = anchor_speed[m]
        weights[m] = (w / w.sum()) * m.mean()
    weights = weights / weights.sum()
    rng = np.random.default_rng(20261031)
    resampled_idx = rng.choice(n0, size=n0, replace=True, p=weights)

    example_to_sample_row = []
    for sample in samples0:
        for j in range(len(sample.state)):
            example_to_sample_row.append((sample, j))

    set_filter_R_convention(0.0, "derived", "quadrature")
    nis_sink = []
    for i in resampled_idx:
        sample, j = example_to_sample_row[i]
        pid = sample.state["player_id"].iloc[j]
        grp = sample.history[sample.history.player_id == pid]
        if len(grp) < 3:
            continue
        _run_filter(grp, Q, use_velocity=False, nis_sink=nis_sink)
    nis_mean_confound = float(np.mean(nis_sink))
    sigma_hat_confound = invert(nis_mean_confound, calib_nis, calib_sigma)
    print(f"true sigma = 0.0 (no sensor noise added, role mix held, speed resampled within role)")
    print(f"nis_mean = {nis_mean_confound:.4f}")
    print(f"sigma_hat = {sigma_hat_confound:.4f} yd  <-- this is the Q/motion-pattern confound, "
         f"expressed in the same units as sensor noise")

    # ================================================================ 3e
    print("\n" + "=" * 70)
    print("3e. End-to-end switch rule vs. oracle")
    print("=" * 70)
    err_df = pd.read_csv(OUTPUT_DIR / "tier1_test_degrade_summary.csv")
    sig_df = pd.read_csv(OUTPUT_DIR / "tier2a_signals_raw.csv")   # has nis_mean per sigma already (frozen R)

    rows = []
    for sigma in TEST_GRID:
        nis_mean = sig_df.loc[sig_df["sigma"] == sigma, "nis_mean"].iloc[0]
        sigma_hat = invert(nis_mean, calib_nis, calib_sigma)

        k5_err = err_df.loc[err_df["sigma"] == sigma, "K5"].iloc[0]
        phys_err = err_df.loc[err_df["sigma"] == sigma, "tbl_kalman_speed"].iloc[0]

        switch_choice = "tbl_kalman_speed" if sigma_hat > CV_CROSSING else "K5"
        switch_err = phys_err if switch_choice == "tbl_kalman_speed" else k5_err

        oracle_choice = "tbl_kalman_speed" if sigma > TEST_CROSSING else "K5"
        oracle_err = phys_err if oracle_choice == "tbl_kalman_speed" else k5_err

        rows.append({"true_sigma": sigma, "sigma_hat": sigma_hat, "k5_err": k5_err, "phys_err": phys_err,
                    "switch_choice": switch_choice, "switch_err": switch_err,
                    "oracle_choice": oracle_choice, "oracle_err": oracle_err})

    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    table.to_csv(OUTPUT_DIR / "tier2a_switch_rule.csv", index=False)

    print(f"\nmean error, always-K5:        {table['k5_err'].mean():.4f}")
    print(f"mean error, always-physics:   {table['phys_err'].mean():.4f}")
    print(f"mean error, switch rule (CV threshold {CV_CROSSING}):  {table['switch_err'].mean():.4f}")
    print(f"mean error, oracle (test crossing {TEST_CROSSING}):    {table['oracle_err'].mean():.4f}")
    print(f"\nswitch rule loses {table['switch_err'].mean() - table['oracle_err'].mean():.4f} yd "
         f"(mean over the grid) to the oracle -- the operational cost of using the "
         f"optimistic CV threshold instead of the true crossing.")

    disagree = table[table["switch_choice"] != table["oracle_choice"]]
    print(f"\ndisagreement between switch rule and oracle at {len(disagree)}/{len(table)} grid points:")
    print(disagree[["true_sigma", "switch_choice", "oracle_choice", "switch_err", "oracle_err"]].to_string(index=False))


if __name__ == "__main__":
    main()
