"""Tier 1 Task N: coverage audit, and picking an honest primary horizon
(TIER1_SPEC_5.md).

    PYTHONPATH=. python scripts/run_tier1_taskn.py  # run from the repo root

`config.MIN_COVERAGE = 0.25` existed from Tier 0a onward but was only ever
wired into plot_error_curves -- every headline *table* in this project,
including the 6.40 yd @3s bar, was reported regardless of how many tracks
actually survived to that horizon. This measures that directly: across
the full 18-week dataset, for every horizon step, how many plays and
tracks have ground truth that far out, and what fraction of the full
sample that is.

Writes outputs/horizon_coverage.csv and outputs/horizon_coverage.png.
Prints the chosen primary horizon: the largest step with >= 50% coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cache_pipeline import build_dataset_cached
from config import DT, MAX_HORIZON, MIN_COVERAGE, OUTPUT_DIR, WEEKS

PRIMARY_HORIZON_COVERAGE = 0.50


def compute_coverage(samples) -> pd.DataFrame:
    """Data-only -- no predictor involved. A play "contributes" at step
    s if its horizon >= s (ground truth exists that far out); a track
    contributing at s is one player-row of such a play.
    """
    horizons = np.array([s.horizon for s in samples])
    tracks_per_play = np.array([len(s.state) for s in samples])
    max_step = int(horizons.max())

    total_plays = len(samples)
    total_tracks = int(tracks_per_play.sum())

    rows = []
    for step in range(1, max_step + 1):
        mask = horizons >= step
        plays = int(mask.sum())
        tracks = int(tracks_per_play[mask].sum())
        rows.append({
            "step": step, "seconds": round(step * DT, 2),
            "plays": plays, "tracks": tracks,
            "coverage_plays": plays / total_plays,
            "coverage_tracks": tracks / total_tracks,
        })
    return pd.DataFrame(rows)


def choose_primary_horizon(coverage: pd.DataFrame, threshold: float) -> dict:
    """Coverage is defined here on TRACKS (what every table actually
    averages over), matching evaluate.py's own units."""
    eligible = coverage[coverage["coverage_tracks"] >= threshold]
    if eligible.empty:
        raise ValueError(f"no horizon step reaches {threshold:.0%} track coverage")
    row = eligible.loc[eligible["step"].idxmax()]
    return row.to_dict()


def plot_coverage(coverage: pd.DataFrame, primary: dict, path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coverage["seconds"], coverage["coverage_tracks"] * 100,
           marker="o", markersize=3, linewidth=1.6, label="track coverage")
    ax.axhline(MIN_COVERAGE * 100, color="0.6", linestyle=":",
              label=f"MIN_COVERAGE ({MIN_COVERAGE:.0%})")
    ax.axhline(PRIMARY_HORIZON_COVERAGE * 100, color="tab:red", linestyle="--",
              label=f"primary-horizon threshold ({PRIMARY_HORIZON_COVERAGE:.0%})")
    ax.axvline(primary["seconds"], color="tab:green", linestyle="--",
              label=f"chosen primary horizon ({primary['seconds']:.1f}s)")
    ax.axvline(3.0, color="tab:orange", linestyle=":",
              label="3s (old headline horizon)")
    ax.set_xlabel("time since throw (s)")
    ax.set_ylabel("track coverage (%)")
    ax.set_title("How much of the sample survives to each horizon")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    print(f"loading {len(WEEKS)} weeks via cache_pipeline...")
    samples, stats = build_dataset_cached(WEEKS)
    print(f"{stats['n_plays']:,} plays, {stats['n_tracks']:,} tracks\n")

    coverage = compute_coverage(samples)
    coverage.to_csv(OUTPUT_DIR / "horizon_coverage.csv", index=False)
    print(f"wrote {OUTPUT_DIR / 'horizon_coverage.csv'}")

    print("\nstep  seconds  plays  tracks  coverage(tracks)")
    for _, row in coverage.iterrows():
        flag = ""
        if row["step"] == 20:
            flag = "  <- 2.0s"
        if row["step"] == 30:
            flag = "  <- 3.0s (old headline)"
        print(f"{int(row['step']):>4}  {row['seconds']:>7.1f}  "
             f"{int(row['plays']):>5}  {int(row['tracks']):>6}  "
             f"{row['coverage_tracks']:>7.1%}{flag}")

    primary = choose_primary_horizon(coverage, PRIMARY_HORIZON_COVERAGE)
    print(f"\nprimary horizon: step {int(primary['step'])} "
         f"({primary['seconds']:.1f}s), {int(primary['plays'])} plays, "
         f"{int(primary['tracks'])} tracks, "
         f"{primary['coverage_tracks']:.1%} track coverage")

    row_3s = coverage[coverage["step"] == 30]
    if not row_3s.empty:
        r = row_3s.iloc[0]
        print(f"3s (secondary): {int(r['plays'])} plays, {int(r['tracks'])} "
             f"tracks, {r['coverage_tracks']:.1%} track coverage")

    plot_coverage(coverage, primary, OUTPUT_DIR / "horizon_coverage.png")


if __name__ == "__main__":
    main()
