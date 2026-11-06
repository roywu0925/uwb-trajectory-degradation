"""Regenerate the K5 / tbl_kalman_speed crossing sigma from a committed-or-local play-level CSV,
with the repo's own estimator, and write the result to a file.

Estimator (copied verbatim from uwb-trajectory-degradation@9bfc7b1):
  - per_play  : run_tier1_k5_tks_crossover.py:110 / run_tier1_test_degrade.py:178
  - _play_sigma_matrix, find_crossing : scripts/run_tier1_crossover.py:110-145
  - bootstrap : 1000 play-level resamples, np.random.default_rng(0), draws where find_crossing
                returns None are DROPPED (as in the repo); n_valid is reported, not hidden.
Usage:
  python scripts/regen_crossing.py PLAYLEVEL.csv --grid 1.0,1.2,1.4,1.6,1.8,2.0 --label cv --out OUT.json
  python scripts/regen_crossing.py --compare CV.json TEST.json --out OPTIMISM.json

Regenerating the committed outputs (the play-level CSVs are gitignored, >2 MB; each output records the
sha256 of the file it was computed from):
  python scripts/regen_crossing.py outputs/tier1_k5_tks_crossover_playlevel.csv \
      --grid 1.0,1.2,1.4,1.6,1.8,2.0 --label cv --out outputs/tier1_cv_crossing_regen.json
  python scripts/regen_crossing.py outputs/tier1_test_crossover_playlevel.csv \
      --grid 0.0,0.8,1.0,1.2,1.4,1.6,1.8,2.0 --label test --out outputs/tier1_test_crossing_regen.json
  python scripts/regen_crossing.py --compare outputs/tier1_cv_crossing_regen.json \
      outputs/tier1_test_crossing_regen.json --out outputs/tier1_crossing_optimism.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_STEP = 19


# ---- verbatim from scripts/run_tier1_crossover.py (9bfc7b1) ----
def find_crossing(sigma_grid, mde_kalman, mde_cv) -> float | None:
    diff = np.asarray(mde_kalman, dtype=float) - np.asarray(mde_cv, dtype=float)
    sigma_grid = np.asarray(sigma_grid, dtype=float)
    for i in range(len(sigma_grid) - 1):
        d0, d1 = diff[i], diff[i + 1]
        if np.isnan(d0) or np.isnan(d1):
            continue
        if d0 == 0:
            return float(sigma_grid[i])
        if np.sign(d0) != np.sign(d1):
            x0, x1 = sigma_grid[i], sigma_grid[i + 1]
            return float(x0 + (0 - d0) * (x1 - x0) / (d1 - d0))
    return None


def _play_sigma_matrix(play_level, sigma_grid, predictor, step):
    per_play = (play_level[(play_level["predictor"] == predictor)
                          & (play_level["step"] == step)]
               .groupby(["game_id", "play_id", "sigma"])["mde"]
               .mean().reset_index())
    plays = sorted(per_play[["game_id", "play_id"]]
                   .drop_duplicates().itertuples(index=False, name=None))
    play_index = {p: i for i, p in enumerate(plays)}
    sigma_index = {s: i for i, s in enumerate(sigma_grid)}
    mat = np.full((len(plays), len(sigma_grid)), np.nan)
    for row in per_play.itertuples(index=False):
        if row.sigma in sigma_index:  # only addition: tolerate sigmas outside the requested grid
            mat[play_index[(row.game_id, row.play_id)], sigma_index[row.sigma]] = row.mde
    return mat, plays
# ----------------------------------------------------------------


def run(path, grid):
    play_level = pd.read_csv(path)
    per_play = (play_level.groupby(["sigma", "seed", "step", "game_id", "play_id", "predictor"])
                ["mde"].mean().reset_index())
    mat_k5, plays = _play_sigma_matrix(per_play, grid, "K5", PRIMARY_STEP)
    mat_tks, _ = _play_sigma_matrix(per_play, grid, "tbl_kalman_speed", PRIMARY_STEP)
    mk5, mtks = np.nanmean(mat_k5, axis=0), np.nanmean(mat_tks, axis=0)
    point = find_crossing(grid, mk5, mtks)
    rng = np.random.default_rng(0)
    n = len(plays)
    draws = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        c = find_crossing(grid, np.nanmean(mat_k5[idx], axis=0), np.nanmean(mat_tks[idx], axis=0))
        draws.append(np.nan if c is None else c)
    d = np.array(draws)
    v = d[~np.isnan(d)]
    n_drop = int(np.isnan(d).sum())
    return {
        "note": (f"{len(v)} of 1000 bootstrap resamples produced a crossing within the sigma grid "
                 f"{grid[0]}-{grid[-1]}; {n_drop} produced none in range and {'was' if n_drop == 1 else 'were'} "
                 "dropped, as in the repo's own bootstrap. The CI is over the valid draws."),
        "point": point,
        "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None,
        "boot_n_valid": int(len(v)), "boot_n_dropped": n_drop,
        "n_plays": n, "n_games": len({g for g, _ in plays}),
        "boot_median": float(np.median(v)) if len(v) else None,
        "grid": list(grid), "k5_mean_by_sigma": [round(float(x), 4) for x in mk5],
        "tks_mean_by_sigma": [round(float(x), 4) for x in mtks],
        "draws": [None if np.isnan(x) else float(x) for x in d],
    }


def compare(cv_path, test_path):
    cv, te = json.loads(Path(cv_path).read_text()), json.loads(Path(test_path).read_text())
    c, t = cv["point"], te["point"]
    dc = np.array([np.nan if x is None else x for x in cv["draws"]])
    dt = np.array([np.nan if x is None else x for x in te["draws"]])
    ok = ~np.isnan(dc) & ~np.isnan(dt)
    o = 100 * (dc[ok] - dt[ok]) / dc[ok]  # independent populations; draws paired by index
    return {
        "cv_point": c, "test_point": t,
        "optimism_pct_of_cv": 100 * (c - t) / c, "optimism_pct_of_test": 100 * (c - t) / t,
        "optimism_boot_median_pct": float(np.median(o)),
        "optimism_boot_ci95_pct": [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))],
        "n_paired_draws": int(ok.sum()),
        "same_estimator": "find_crossing + 1000 play-level resamples, default_rng(0), step 19, both sides",
        "differs": {"cv_grid": cv["grid"], "test_grid": te["grid"],
                    "note": "CV play-level data has no sigma < 1.0, so the test crossing (0.93) cannot be "
                            "estimated on the CV grid; dropping the test grid's 1.0 point moves the test "
                            "crossing by +0.0068, bounding the grid effect at ~0.01 of the 0.12 gap"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("playlevel", nargs="?")
    ap.add_argument("--grid")
    ap.add_argument("--label")
    ap.add_argument("--compare", nargs=2, metavar=("CV_JSON", "TEST_JSON"))
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.compare:
        r = compare(*a.compare)
        print(json.dumps(r, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps(r, indent=1))
        return
    grid = [float(x) for x in a.grid.split(",")]
    r = {"label": a.label, **run(a.playlevel, grid),
         "source_file": Path(a.playlevel).name,
         "source_sha256": hashlib.sha256(Path(a.playlevel).read_bytes()).hexdigest()}
    summ = {k: v for k, v in r.items() if k != "draws"}
    print(json.dumps(summ, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
