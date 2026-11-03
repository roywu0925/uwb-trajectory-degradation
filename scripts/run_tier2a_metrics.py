"""Tier 2A, metrics A and B, read off the already-published test sweep
and the saved signal table from run_tier2a_drift.py. Produces no new
test claim for metric A (interpolates published numbers only). Metric B
builds and verifies a benign shift, then checks each signal against it.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import torch

from baselines import BASELINES
from config import OUTPUT_DIR, WEEKS
from data_loader import add_kinematics, build_samples, load_raw, normalise
from kalman import H_POS, _run_filter, build_Q
from model import GRUResidualModel, build_examples
from run_tier1_taskk import build_tensors, predict_all
from run_tier1_taskp import build_predictors_quadrature
from run_tier1_r_convention_check import set_filter_R_convention
from run_tier2a_drift import FEATURE_NAMES, PSI_ALARM, SPC_K, decile_edges, psi

PRIMARY_STEP = 19
COL = PRIMARY_STEP - 1


def interp_crossing(sigma_grid, values, threshold):
    """First sigma (linear interp) where `values` crosses above
    `threshold`. None if it never does, 0.0-boundary if already above
    at the first grid point (reported explicitly, not hidden)."""
    sigma_grid = np.asarray(sigma_grid, dtype=float)
    values = np.asarray(values, dtype=float)
    diff = values - threshold
    if diff[0] > 0:
        return sigma_grid[0], "already above threshold at the first grid point"
    for i in range(len(sigma_grid) - 1):
        d0, d1 = diff[i], diff[i + 1]
        if d0 <= 0 and d1 > 0:
            x0, x1 = sigma_grid[i], sigma_grid[i + 1]
            c = x0 + (0 - d0) * (x1 - x0) / (d1 - d0)
            return float(c), None
    return None, "never crosses in tested range"


def main():
    sig_df = pd.read_csv(OUTPUT_DIR / "tier2a_signals_raw.csv")
    with open(OUTPUT_DIR / "tier2a_state.pkl", "rb") as f:
        state = pickle.load(f)
    err_df = pd.read_csv(OUTPUT_DIR / "tier1_test_degrade_summary.csv")

    sigma_grid = sig_df["sigma"].to_numpy()
    assert np.allclose(sigma_grid, err_df["sigma"].to_numpy()), "grids must match for metric A"

    print("=" * 70)
    print("METRIC A -- lead time (soft / hard), read off already-published numbers")
    print("=" * 70)

    results = {}
    for model in ("K3", "K5"):
        err = err_df[model].to_numpy()
        err0 = err[0]
        soft_thresh = err0 * 1.10
        best_phys = err_df[["constant_velocity", "tbl_kalman_speed", "velocity_turning_to_ball"]].min(axis=1).to_numpy()

        soft_fail, soft_note = interp_crossing(sigma_grid, err, soft_thresh)
        hard_fail, hard_note = interp_crossing(sigma_grid, err - best_phys, 0.0)
        print(f"\n{model}: sigma=0 error={err0:.4f}, soft threshold (+10%)={soft_thresh:.4f}")
        print(f"  soft failure sigma: {soft_fail} ({soft_note})")
        print(f"  hard failure sigma (vs best of constant_velocity/tbl_kalman_speed/velocity_turning_to_ball): "
             f"{hard_fail} ({hard_note})")
        results[model] = {"soft_fail": soft_fail, "hard_fail": hard_fail}

    # ---- alarm sigmas per signal
    print("\n" + "=" * 70)
    print("Alarm sigmas per signal")
    print("=" * 70)

    psi_max = sig_df["psi_max"].to_numpy()
    sig1_alarm, sig1_note = interp_crossing(sigma_grid, psi_max, PSI_ALARM)
    print(f"signal 1 (input PSI max > {PSI_ALARM}): alarm sigma = {sig1_alarm} ({sig1_note}) -- model-independent")

    nis_mean = sig_df["nis_mean"].to_numpy()
    nis_n = sig_df["nis_n"].to_numpy()
    dof = state["dof"]
    z = 1.959964
    nis_threshold = dof + z * np.sqrt(2 * dof / nis_n)
    print(f"signal 4 theoretical threshold per sigma (dof={dof}, N varies {nis_n.min()}-{nis_n.max()}): "
         f"{np.round(nis_threshold, 4)}")
    print(f"signal 4 nis_mean: {np.round(nis_mean, 4)}")
    sig4_alarm, sig4_note = interp_crossing(sigma_grid, nis_mean - nis_threshold, 0.0)
    print(f"signal 4 (batch-mean NIS > theoretical bound): alarm sigma = {sig4_alarm} ({sig4_note}) -- model-independent")

    alarm = {"signal1_input_psi": {"K3": sig1_alarm, "K5": sig1_alarm},
            "signal4_nis": {"K3": sig4_alarm, "K5": sig4_alarm}}

    for model in ("K3", "K5"):
        baseline_mean, baseline_sd = state["sig2_baseline"][model]
        thresh2 = baseline_mean + SPC_K * baseline_sd
        vals2 = sig_df[f"sig2_{model}_mean"].to_numpy()
        a2, n2 = interp_crossing(sigma_grid, vals2, thresh2)
        print(f"signal 2 ({model}, threshold={thresh2:.3f} = train mean+{SPC_K}sd): alarm sigma = {a2} ({n2})")

        vals3 = sig_df[f"sig3_{model}_psi"].to_numpy()
        a3, n3 = interp_crossing(sigma_grid, vals3, PSI_ALARM)
        print(f"signal 3 ({model}, output PSI > {PSI_ALARM}): alarm sigma = {a3} ({n3})")

        alarm.setdefault("signal2_model_physics", {})[model] = a2
        alarm.setdefault("signal3_output_psi", {})[model] = a3

    print("\n" + "=" * 70)
    print("Signal 3 vs signal 1 redundancy check")
    print("=" * 70)
    for model in ("K3", "K5"):
        c = np.corrcoef(sig_df["psi_max"], sig_df[f"sig3_{model}_psi"])[0, 1]
        print(f"corr(signal1_psi_max, signal3_{model}_psi) across the sweep = {c:.4f}")

    # ---- lead times
    print("\n" + "=" * 70)
    print("DELIVERABLE TABLE -- lead times")
    print("=" * 70)
    table_rows = []
    for sig_name, thresh_source in (
        ("1. input PSI max", "convention (PSI>0.2)"),
        ("2. model-vs-physics distance", "training data (mean+3sd)"),
        ("3. output PSI", "convention (PSI>0.2)"),
        ("4. Kalman NIS (frozen R)", "theory (chi2, dof=2)"),
    ):
        for model in ("K3", "K5"):
            key = {"1. input PSI max": "signal1_input_psi",
                  "2. model-vs-physics distance": "signal2_model_physics",
                  "3. output PSI": "signal3_output_psi",
                  "4. Kalman NIS (frozen R)": "signal4_nis"}[sig_name]
            a = alarm[key][model]
            soft_f = results[model]["soft_fail"]
            hard_f = results[model]["hard_fail"]
            lead_soft = (soft_f - a) if (a is not None and soft_f is not None) else None
            lead_hard = (hard_f - a) if (a is not None and hard_f is not None) else None
            table_rows.append({
                "signal": sig_name, "model": model, "threshold_source": thresh_source,
                "alarm_sigma": a, "lead_soft": lead_soft, "lead_hard": lead_hard,
            })
    table = pd.DataFrame(table_rows)
    print(table.to_string(index=False))
    table.to_csv(OUTPUT_DIR / "tier2a_leadtime_table.csv", index=False)

    # ================================================================ Metric B
    print("\n" + "=" * 70)
    print("METRIC B -- benign shift (resample toward higher true speed), sigma=0")
    print("=" * 70)

    with open(OUTPUT_DIR / "splits.json") as f:
        splits = json.load(f)
    test_games = set(splits["test"])
    inp, out = load_raw(weeks=WEEKS)
    inp = inp[inp["game_id"].isin(test_games)].reset_index(drop=True)
    out = out[out["game_id"].isin(test_games)].reset_index(drop=True)
    inp, out = normalise(inp, out)
    inp = add_kinematics(inp)
    samples0 = build_samples(inp, out)
    examples0 = build_examples(samples0)
    tensors0 = build_tensors(examples0)
    n0 = len(examples0)
    anchor0 = tensors0["anchor"].numpy()
    base_anchor0 = np.repeat(anchor0[:, None, :], tensors0["truth"].shape[1], axis=1)
    base_t0 = torch.from_numpy(base_anchor0)
    truth0 = tensors0["truth"].numpy()
    mask0 = tensors0["mask"].numpy()
    hist_padded0 = tensors0["hist_padded"].numpy()
    hist_lengths0 = tensors0["hist_lengths"].numpy()

    anchor_speed = np.array([np.hypot(hist_padded0[i, hist_lengths0[i]-1, 2],
                                      hist_padded0[i, hist_lengths0[i]-1, 3])
                             for i in range(n0)])
    print(f"anchor speed: mean={anchor_speed.mean():.3f} median={np.median(anchor_speed):.3f} "
         f"max={anchor_speed.max():.3f}")

    weights = anchor_speed / anchor_speed.sum()
    rng = np.random.default_rng(20261031)
    resampled_idx = rng.choice(n0, size=n0, replace=True, p=weights)
    print(f"resampled anchor speed: mean={anchor_speed[resampled_idx].mean():.3f} "
         f"(full population: {anchor_speed.mean():.3f}) -- confirms the input shift happened")

    k3 = GRUResidualModel(hidden=torch.load(OUTPUT_DIR / "checkpoints" / "K3_FINAL_fold0.pt", weights_only=False)["config"]["hidden"])
    k3.load_state_dict(torch.load(OUTPUT_DIR / "checkpoints" / "K3_FINAL_fold0.pt", weights_only=False)["model_state_dict"])
    k3.eval()
    k5 = GRUResidualModel(hidden=torch.load(OUTPUT_DIR / "checkpoints" / "K5_FINAL_fold0.pt", weights_only=False)["config"]["hidden"])
    k5.load_state_dict(torch.load(OUTPUT_DIR / "checkpoints" / "K5_FINAL_fold0.pt", weights_only=False)["model_state_dict"])
    k5.eval()

    valid_full = mask0[:, COL] > 0
    benign_report = {}
    for name, model in (("K3", k3), ("K5", k5)):
        model_out_full = predict_all(model, base_t0, tensors0, np.arange(n0))
        pred_full = base_anchor0 + model_out_full
        err_full = np.linalg.norm(pred_full[valid_full, COL] - truth0[valid_full, COL], axis=-1)

        # per-track error, then index into resampled_idx (restricted to valid tracks' own indices)
        err_per_track = np.full(n0, np.nan)
        err_per_track[valid_full] = err_full
        resampled_err = err_per_track[resampled_idx]
        resampled_err = resampled_err[~np.isnan(resampled_err)]

        boot_means = []
        rng2 = np.random.default_rng(0)
        valid_idx = np.where(valid_full)[0]
        for _ in range(1000):
            bidx = rng2.choice(valid_idx, size=len(valid_idx), replace=True)
            boot_means.append(err_per_track[bidx].mean())
        boot_means = np.array(boot_means)
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        print(f"\n{name}: full-population error={err_full.mean():.4f}, "
             f"95% bootstrap CI=({ci_lo:.4f}, {ci_hi:.4f}); "
             f"resampled (speed-weighted) error={resampled_err.mean():.4f}")
        benign = ci_lo <= resampled_err.mean() <= ci_hi
        print(f"  benign? {'YES' if benign else 'NO'} (resampled mean "
             f"{'inside' if benign else 'outside'} the full population's own bootstrap CI)")
        benign_report[name] = benign

    if not all(benign_report.values()):
        print("\nAt least one model's error shifted under this resampling -- "
             "the construction is NOT benign for that model. Reporting as such, "
             "not substituting a different construction without flagging it.")

    # ---- does each signal fire on the (benign-or-not) resampled population?
    print("\nSignal values on the resampled population vs. their alarm thresholds:")
    feature_pool_resampled = {}
    for c, fname in enumerate(FEATURE_NAMES):
        vals = []
        for i in resampled_idx:
            L = hist_lengths0[i]
            vals.append(hist_padded0[i, :L, c])
        feature_pool_resampled[fname] = np.concatenate(vals)
    psi_resampled = {fname: psi(feature_pool_resampled[fname], state["ref_feature_pool"][fname],
                               state["ref_edges"][fname]) for fname in FEATURE_NAMES}
    psi_max_resampled = max(psi_resampled.values())
    print(f"  signal 1 (input PSI max): {psi_max_resampled:.4f} vs threshold {PSI_ALARM} "
         f"-> {'FIRES' if psi_max_resampled > PSI_ALARM else 'does not fire'}")

    set_filter_R_convention(0.0, "derived", "quadrature")
    dof = state["dof"]
    nis_sink = []
    # NIS needs the raw history group per track -- map example index back to (sample, row)
    pos = 0
    example_to_sample_row = []
    for sample in samples0:
        n = len(sample.state)
        for j in range(n):
            example_to_sample_row.append((sample, j))
        pos += n
    Q = build_Q(state["sigma_a"])
    for i in resampled_idx:
        sample, j = example_to_sample_row[i]
        player_id = sample.state["player_id"].iloc[j]
        grp = sample.history[sample.history.player_id == player_id]
        if len(grp) < 3:
            continue
        _run_filter(grp, Q, use_velocity=False, nis_sink=nis_sink)
    nis_mean_resampled = float(np.mean(nis_sink))
    z = 1.959964
    nis_thresh_resampled = dof + z * np.sqrt(2 * dof / len(nis_sink))
    print(f"  signal 4 (NIS, frozen R): {nis_mean_resampled:.4f} vs threshold {nis_thresh_resampled:.4f} "
         f"-> {'FIRES' if nis_mean_resampled > nis_thresh_resampled else 'does not fire'}")

    for name, model in (("K3", k3), ("K5", k5)):
        set_filter_R_convention(0.0, "derived", "quadrature")
        predictors_clean = build_predictors_quadrature(state["sigma_a"])
        base_tks0 = np.zeros_like(base_anchor0)
        pos = 0
        for sample in samples0:
            n = len(sample.state)
            base_tks0[pos:pos+n] = predictors_clean["tbl_kalman_speed"](sample, tensors0["truth"].shape[1])
            pos += n

        model_out_resampled = predict_all(model, torch.from_numpy(base_anchor0), tensors0, resampled_idx)
        pred_resampled = base_anchor0[resampled_idx] + model_out_resampled
        dist_resampled = np.linalg.norm(pred_resampled[:, COL] - base_tks0[resampled_idx, COL], axis=-1)
        baseline_mean, baseline_sd = state["sig2_baseline"][name]
        thresh2 = baseline_mean + SPC_K * baseline_sd
        print(f"  signal 2 ({name}): {dist_resampled.mean():.4f} vs threshold {thresh2:.4f} "
             f"-> {'FIRES' if dist_resampled.mean() > thresh2 else 'does not fire'}")

        disp_mag_resampled = np.linalg.norm(model_out_resampled[:, COL], axis=-1)
        psi3 = psi(disp_mag_resampled, state["sig3_ref_pool"][name], decile_edges(state["sig3_ref_pool"][name]))
        print(f"  signal 3 ({name}): {psi3:.4f} vs threshold {PSI_ALARM} "
             f"-> {'FIRES' if psi3 > PSI_ALARM else 'does not fire'}")


if __name__ == "__main__":
    main()
