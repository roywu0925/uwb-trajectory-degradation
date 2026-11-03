"""Tier 1 Task J: expand the data, build honest splits (TIER1_SPEC_4.md).

    python run_tier1_taskj.py

Loads all 18 weeks via cache_pipeline (one week at a time, cached to
data/cache/), reports peak memory and load time, builds a fixed 70/15/15
train/val/test split BY GAME (not by play), and writes the game IDs to
outputs/splits.json so every later run -- including Task K/L's model --
uses the identical partition.
"""

from __future__ import annotations

import json
import platform
import resource
import sys

import numpy as np
import pandas as pd

from cache_pipeline import build_dataset_cached
from config import DT, OUTPUT_DIR, SPLIT_FRACTIONS, SPLIT_SEED, WEEKS

STEP_3S = round(3.0 / DT)


def peak_memory_mb() -> float:
    """Peak resident set size so far, in MB. ru_maxrss is bytes on
    macOS/Darwin, kilobytes on Linux -- this process only needs to run
    correctly on the platform it's actually run on, but the unit is
    worth stating rather than silently getting a Linux run 1000x wrong.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return raw / divisor


def build_splits(samples, seed: int, fractions: dict) -> dict:
    game_ids = sorted({s.game_id for s in samples})
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(game_ids)
    n = len(shuffled)
    n_train = round(n * fractions["train"])
    n_val = round(n * fractions["val"])
    return {
        "train": sorted(int(g) for g in shuffled[:n_train]),
        "val": sorted(int(g) for g in shuffled[n_train:n_train + n_val]),
        "test": sorted(int(g) for g in shuffled[n_train + n_val:]),
    }


def split_stats(samples, splits: dict) -> pd.DataFrame:
    rows = []
    for name, game_ids in splits.items():
        game_set = set(game_ids)
        sub = [s for s in samples if s.game_id in game_set]
        n_plays = len(sub)
        n_tracks = sum(len(s.state) for s in sub)
        n_tracks_3s = sum(len(s.state) for s in sub if s.horizon >= STEP_3S)
        rows.append({"split": name, "games": len(game_ids), "plays": n_plays,
                    "tracks": n_tracks, "tracks_at_3s": n_tracks_3s})
    return pd.DataFrame(rows)


def main():
    print(f"loading {len(WEEKS)} weeks via cache_pipeline...\n")
    samples, stats = build_dataset_cached(WEEKS)

    peak_mb = peak_memory_mb()
    print(f"\ntotal load time: {stats['total_seconds']:.1f}s")
    print(f"peak memory (RSS, this process): {peak_mb:.0f} MB "
         f"[{platform.system()}, Python {sys.version.split()[0]}]")
    print(f"dataset: {stats['n_plays']:,} plays, {stats['n_tracks']:,} "
         f"player-tracks across {len(WEEKS)} weeks")

    n_games = len({s.game_id for s in samples})
    print(f"{n_games} unique games")

    splits = build_splits(samples, SPLIT_SEED, SPLIT_FRACTIONS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "splits.json", "w") as f:
        json.dump({"seed": SPLIT_SEED, "fractions": SPLIT_FRACTIONS,
                  **splits}, f, indent=2)
    print(f"\nwrote {OUTPUT_DIR / 'splits.json'}")

    table = split_stats(samples, splits)
    print("\nsplit counts:")
    print(table.to_string(index=False))

    overlap = (set(splits["train"]) & set(splits["val"])
              | set(splits["train"]) & set(splits["test"])
              | set(splits["val"]) & set(splits["test"]))
    assert not overlap, f"game ID overlap between splits: {overlap}"
    assert sum(len(v) for v in splits.values()) == n_games, \
        "split game counts don't sum to the total -- a game got dropped or duplicated"
    print("\nno game ID overlap between splits; all games accounted for")


if __name__ == "__main__":
    main()
