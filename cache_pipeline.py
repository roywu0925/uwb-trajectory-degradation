"""Per-week caching so the full 18-week dataset never needs to sit in
memory as raw CSV/DataFrames all at once (TIER1_SPEC_4.md, Task J).

Each week's input/output is loaded once (usecols-trimmed --
data_loader.load_raw_week), pushed through the same
normalise -> add_kinematics -> build_samples pipeline as
data_loader.build_dataset, and the resulting PlaySamples are written to
data/cache/<week>/ as parquet (the three DataFrame fields: state,
history, context) plus one compressed .npz (the truth arrays, one per
play, keyed by "<game_id>_<play_id>"). `horizon` isn't stored
separately -- it's truth.shape[1], recovered for free on reload. The
raw frame for that week is freed (del + gc.collect) before the next
week is loaded, so peak memory is one week's CSVs, not eighteen.

On a later run, build_dataset_cached reads straight from these files
instead of re-parsing CSV, which is where the caching actually pays for
itself -- Tier 0/1 scripts get re-run often, the raw files don't change.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, WEEKS
from data_loader import (PlaySample, add_kinematics, build_samples,
                         load_raw_week, normalise)

CACHE_DIR = DATA_DIR / "cache"

_TAG_COLS = ["__game_id", "__play_id"]


def _week_cache_dir(week: str) -> Path:
    return CACHE_DIR / week


def is_cached(week: str) -> bool:
    d = _week_cache_dir(week)
    return all((d / name).exists() for name in
              ("state.parquet", "history.parquet", "context.parquet", "truth.npz"))


def _tag(df: pd.DataFrame, game_id: int, play_id: int) -> pd.DataFrame:
    df = df.copy()
    df["__game_id"] = game_id
    df["__play_id"] = play_id
    return df


def save_week_cache(week: str, samples: list[PlaySample]) -> None:
    d = _week_cache_dir(week)
    d.mkdir(parents=True, exist_ok=True)

    state_parts, history_parts, context_parts = [], [], []
    truth_arrays = {}
    for s in samples:
        state_parts.append(_tag(s.state, s.game_id, s.play_id))
        history_parts.append(_tag(s.history, s.game_id, s.play_id))
        context_parts.append(_tag(s.context, s.game_id, s.play_id))
        truth_arrays[f"{s.game_id}_{s.play_id}"] = s.truth

    pd.concat(state_parts, ignore_index=True).to_parquet(d / "state.parquet")
    pd.concat(history_parts, ignore_index=True).to_parquet(d / "history.parquet")
    pd.concat(context_parts, ignore_index=True).to_parquet(d / "context.parquet")
    np.savez_compressed(d / "truth.npz", **truth_arrays)


def load_week_cache(week: str) -> list[PlaySample]:
    d = _week_cache_dir(week)
    state = pd.read_parquet(d / "state.parquet")
    history = pd.read_parquet(d / "history.parquet")
    context = pd.read_parquet(d / "context.parquet")
    truth_npz = np.load(d / "truth.npz")

    state_groups = dict(tuple(state.groupby(_TAG_COLS, sort=False)))
    history_groups = dict(tuple(history.groupby(_TAG_COLS, sort=False)))
    context_groups = dict(tuple(context.groupby(_TAG_COLS, sort=False)))

    samples = []
    for key in truth_npz.files:
        gid_s, pid_s = key.split("_")
        gid, pid = int(gid_s), int(pid_s)
        truth = truth_npz[key]
        samples.append(PlaySample(
            game_id=gid, play_id=pid,
            state=state_groups[(gid, pid)].drop(columns=_TAG_COLS).reset_index(drop=True),
            history=history_groups[(gid, pid)].drop(columns=_TAG_COLS).reset_index(drop=True),
            truth=truth, horizon=truth.shape[1],
            context=context_groups[(gid, pid)].drop(columns=_TAG_COLS).reset_index(drop=True)))
    return samples


def build_week_cache(week: str) -> list[PlaySample]:
    """Load one week's raw CSVs, run it through the full pipeline, cache
    the result, and free the raw frame before returning."""
    inp, out = load_raw_week(week)
    inp, out = normalise(inp, out)
    inp = add_kinematics(inp)
    samples = build_samples(inp, out)
    save_week_cache(week, samples)
    del inp, out
    gc.collect()
    return samples


def build_dataset_cached(weeks: list[str] | None = None,
                         force_rebuild: bool = False
                         ) -> tuple[list[PlaySample], dict]:
    """The multi-week equivalent of data_loader.build_dataset, but one
    week at a time: whatever's already cached is read from disk,
    whatever isn't is built (and cached) and then freed before the next
    week. Returns (samples, stats) -- stats has per-week timing/counts
    for the memory/time report (Task J).
    """
    weeks = WEEKS if weeks is None else weeks
    all_samples: list[PlaySample] = []
    per_week = []
    t_start = time.time()
    for week in weeks:
        t0 = time.time()
        if not force_rebuild and is_cached(week):
            samples = load_week_cache(week)
            source = "cache"
        else:
            samples = build_week_cache(week)
            source = "built"
        dt = time.time() - t0
        n_tracks = sum(len(s.state) for s in samples)
        per_week.append({"week": week, "n_plays": len(samples),
                         "n_tracks": n_tracks, "source": source,
                         "seconds": round(dt, 2)})
        print(f"  {week}: {len(samples)} plays, {n_tracks} tracks "
             f"({source}, {dt:.2f}s)")
        all_samples.extend(samples)
        del samples
        gc.collect()

    stats = {"per_week": per_week, "total_seconds": round(time.time() - t_start, 2),
             "n_plays": len(all_samples),
             "n_tracks": sum(len(s.state) for s in all_samples)}
    return all_samples, stats
