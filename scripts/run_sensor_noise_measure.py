"""Measure NGS's actual sensor noise from stationary pre-snap players,
instead of using the three assumed/uncited constants this project has
carried so far.

SELECTOR, chosen because the data forces it (see chat for the full
reasoning):
  - No event column exists in this dataset at all (checked: input CSV's
    23 columns, none of them an event/line_set/ball_snap/motion marker).
  - No offensive line positions exist either (checked: player_position's
    full value set is {CB,DE,DT,FB,FS,ILB,MLB,NT,OLB,QB,RB,S,SS,TE,WR} --
    this competition's data excludes linemen entirely).
  - Substitute: player_role == "Passer" (the QB) as the "most reliably
    still" role actually available, since every play has exactly one and
    pre-snap the QB is stationary in essentially every formation.
  - Temporal window: frame_id <= K (the EARLIEST frames of each play's
    tracked history), K chosen small (5) to minimise the chance of
    capturing snap/dropback motion, with K=3 and K=8 as a sensitivity
    check -- not tuned to produce a particular answer, reported for
    robustness.
  - Selection is on player_role and frame_id only -- never on x, y, s,
    dir, or any derived position/speed statistic.

Both the QB-only estimate and the broader (all players, same window)
estimate are reported, per instruction.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, WEEKS
from data_loader import TRAIN_DIR

K_WINDOWS = [5, 3, 8]
MIN_SEGMENT_LEN = 3


def load_raw_filtered(games: set) -> pd.DataFrame:
    cols = ["game_id", "play_id", "nfl_id", "frame_id", "player_role", "x", "y"]
    parts = []
    for week in WEEKS:
        for path in sorted(TRAIN_DIR.glob(f"input_*_{week}.csv")):
            df = pd.read_csv(path, usecols=cols)
            df = df[df["game_id"].isin(games)]
            if not df.empty:
                parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build_segments(df: pd.DataFrame, k: int, role: str | None):
    sub = df[df["frame_id"] <= k]
    if role is not None:
        sub = sub[sub["player_role"] == role]
    segments = []
    for (gid, pid, nid), grp in sub.groupby(["game_id", "play_id", "nfl_id"], sort=False):
        grp = grp.sort_values("frame_id")
        if len(grp) < MIN_SEGMENT_LEN:
            continue
        segments.append({"game_id": gid, "play_id": pid, "nfl_id": nid,
                         "x": grp["x"].to_numpy(), "y": grp["y"].to_numpy()})
    return segments


def successive_diff_sigma(segments, axis):
    diffs = []
    for seg in segments:
        v = seg[axis]
        diffs.append(np.diff(v))
    diffs = np.concatenate(diffs) if diffs else np.array([])
    return float(np.std(diffs) / np.sqrt(2)), diffs


def detrend_residuals(segments, axis):
    residuals_per_segment = []
    for seg in segments:
        v = seg[axis]
        t = np.arange(len(v))
        if len(v) < 3:
            continue
        coeffs = np.polyfit(t, v, 1)
        fit = np.polyval(coeffs, t)
        residuals_per_segment.append(v - fit)
    return residuals_per_segment


def detrended_sigma(residuals_per_segment):
    all_res = np.concatenate(residuals_per_segment) if residuals_per_segment else np.array([])
    return float(np.std(all_res)), all_res


def acf_pooled(residuals_per_segment, max_lag=10):
    """Pearson correlation at each lag, pairing only WITHIN a segment,
    pooled across all segments -- never across segment boundaries."""
    out = []
    for lag in range(1, max_lag + 1):
        a_vals, b_vals = [], []
        for res in residuals_per_segment:
            if len(res) > lag:
                a_vals.append(res[:-lag])
                b_vals.append(res[lag:])
        if not a_vals:
            out.append(np.nan)
            continue
        a = np.concatenate(a_vals)
        b = np.concatenate(b_vals)
        if a.std() < 1e-10 or b.std() < 1e-10:
            out.append(np.nan)
        else:
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return out


def bootstrap_sigma(segments, axis, method, n_resamples=1000, seed=0):
    """Play-level grouped bootstrap -- resample (game_id, play_id) units
    with replacement, consistent with the rest of this project's
    bootstrap convention."""
    play_keys = sorted(set((s["game_id"], s["play_id"]) for s in segments))
    by_play = {}
    for s in segments:
        by_play.setdefault((s["game_id"], s["play_id"]), []).append(s)

    rng = np.random.default_rng(seed)
    n_plays = len(play_keys)
    estimates = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n_plays, size=n_plays)
        resampled_segments = []
        for i in idx:
            resampled_segments.extend(by_play[play_keys[i]])
        if method == "diff":
            val, _ = successive_diff_sigma(resampled_segments, axis)
        else:
            res = detrend_residuals(resampled_segments, axis)
            val, _ = detrended_sigma(res)
        estimates.append(val)
    return np.array(estimates)


def report_for_population(df, k, role, label, bootstrap=True):
    print(f"\n--- {label}, K={k} ---")
    segments = build_segments(df, k, role)
    n_segments = len(segments)
    n_plays = len(set((s["game_id"], s["play_id"]) for s in segments))
    print(f"segments={n_segments} plays={n_plays}")
    if n_segments < 10:
        print("too few segments, skipping")
        return None

    results = {}
    for axis in ("x", "y"):
        diff_sigma, _ = successive_diff_sigma(segments, axis)
        res_per_seg = detrend_residuals(segments, axis)
        det_sigma, all_res = detrended_sigma(res_per_seg)
        disagreement = abs(diff_sigma - det_sigma) / ((diff_sigma + det_sigma) / 2) * 100
        print(f"  {axis}: successive-diff sigma_hat={diff_sigma:.4f}  "
             f"detrended-residual sigma_hat={det_sigma:.4f}  disagreement={disagreement:.1f}%")
        results[axis] = {"diff": diff_sigma, "detrend": det_sigma, "disagreement_pct": disagreement,
                         "res_per_seg": res_per_seg}

    # pooled (both axes concatenated)
    diffs_pooled = np.concatenate([successive_diff_sigma(segments, "x")[1],
                                   successive_diff_sigma(segments, "y")[1]])
    res_pooled_per_seg = detrend_residuals(segments, "x") + detrend_residuals(segments, "y")
    pooled_diff_sigma = float(np.std(diffs_pooled) / np.sqrt(2))
    pooled_det_sigma, _ = detrended_sigma(res_pooled_per_seg)
    pooled_disagreement = abs(pooled_diff_sigma - pooled_det_sigma) / ((pooled_diff_sigma + pooled_det_sigma) / 2) * 100
    print(f"  pooled (x+y): successive-diff sigma_hat={pooled_diff_sigma:.4f}  "
         f"detrended-residual sigma_hat={pooled_det_sigma:.4f}  disagreement={pooled_disagreement:.1f}%")

    acf = acf_pooled(res_pooled_per_seg, max_lag=10)
    print(f"  ACF of detrended residuals, lags 1-10: {[round(a,4) if a==a else None for a in acf]}")

    if bootstrap:
        boot_diff = bootstrap_sigma(segments, "x", "diff")   # representative axis for CI width
        ci = np.percentile(boot_diff, [2.5, 97.5])
        print(f"  bootstrap 95% CI (x, successive-diff, play-level): ({ci[0]:.4f}, {ci[1]:.4f})")

    return {"pooled_diff_sigma": pooled_diff_sigma, "pooled_det_sigma": pooled_det_sigma,
           "pooled_disagreement_pct": pooled_disagreement, "acf": acf,
           "per_axis": results, "n_segments": n_segments, "n_plays": n_plays}


def main():
    with open(OUTPUT_DIR / "final_val_games.json") as f:
        fv = json.load(f)
    train_games = set(int(g) for g in fv["final_train_games"]) | set(int(g) for g in fv["final_val_games"])
    with open(OUTPUT_DIR / "splits.json") as f:
        splits = json.load(f)
    test_games = set(splits["test"])

    print("loading raw tracking data, training games only...")
    train_raw = load_raw_filtered(train_games)
    print(f"{len(train_raw):,} rows, {train_raw['game_id'].nunique()} games")

    all_results = {}
    for k in K_WINDOWS:
        all_results[("QB", k)] = report_for_population(train_raw, k, "Passer", f"TRAINING, QB only")
        all_results[("ALL", k)] = report_for_population(train_raw, k, None, f"TRAINING, all roles")

    print("\n" + "=" * 70)
    print("TEST GAMES -- consistency check only, no decision attached")
    print("=" * 70)
    print("loading raw tracking data, test games only...")
    test_raw = load_raw_filtered(test_games)
    print(f"{len(test_raw):,} rows, {test_raw['game_id'].nunique()} games")
    for k in K_WINDOWS:
        report_for_population(test_raw, k, "Passer", f"TEST, QB only", bootstrap=False)
        report_for_population(test_raw, k, None, f"TEST, all roles", bootstrap=False)


if __name__ == "__main__":
    main()
