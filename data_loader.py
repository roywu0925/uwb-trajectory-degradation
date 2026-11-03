"""Load the Big Data Bowl input/output pairs and turn them into samples.

One sample = one play:

    state  : DataFrame, one row per player_to_predict, at the last frame
             before the throw (the anchor)
    history: DataFrame, every pre-throw frame (frame_id, player_id, x, y,
             vx, vy) for those same players, oldest first -- what Tier 0b's
             Kalman filter runs predict/update over. The other baselines
             ignore it and only touch `state`.
    truth  : ndarray (n_players, horizon, 2), the true positions while the
             ball is in the air
    horizon: how many frames this play asks for (num_frames_output)

The competition has already done the hard part -- the input file ends at
the throw and the output file begins there -- so there is no event column
to hunt for and no anchor to guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (DT, FIELD_LENGTH, FIELD_WIDTH, MAX_HORIZON, MAX_PLAYS,
                    MIN_HORIZON, ROLES_OF_INTEREST, TRAIN_DIR, WEEKS)

KEY = ["game_id", "play_id"]

RENAME = {
    "nfl_id": "player_id",
    "play_direction": "play_dir",
    "player_position": "position",
    "player_side": "side",
    "player_role": "role",
    "s": "speed",
    "a": "accel",
    "dir": "direction",
    "o": "orientation",
}

# Columns this pipeline actually reads, pre-rename. player_name,
# player_height, player_weight, player_birth_date, player_position,
# absolute_yardline_number and num_frames_output are present in the raw
# files but never used downstream (horizon comes from the output file's
# own frame count, not num_frames_output); o (orientation) is mirrored
# in normalise() but nothing reads it afterwards. At 18 weeks these are
# most of the memory: object/string columns cost far more per cell than
# the floats actually needed (TIER1_SPEC_4.md, Task J).
INPUT_USECOLS = ["game_id", "play_id", "player_to_predict", "nfl_id",
                 "frame_id", "play_direction", "player_role", "x", "y",
                 "s", "a", "dir", "ball_land_x", "ball_land_y"]
OUTPUT_USECOLS = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y"]


@dataclass
class PlaySample:
    game_id: int
    play_id: int
    state: pd.DataFrame      # players to predict, at the anchor frame
    history: pd.DataFrame    # players to predict, every pre-throw frame
    truth: np.ndarray        # (n_players, horizon, 2)
    horizon: int
    context: pd.DataFrame    # every player at the anchor frame, for plots

    @property
    def key(self):
        return (self.game_id, self.play_id)


# ------------------------------------------------------------------ load

def _week_files(prefix: str, weeks: list | None = None) -> list:
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"{TRAIN_DIR} not found -- put the Kaggle train/ folder there.")
    weeks = WEEKS if weeks is None else weeks
    if weeks:
        paths = []
        for week in weeks:
            found = sorted(TRAIN_DIR.glob(f"{prefix}_*_{week}.csv"))
            if not found:
                print(f"  warning: no {prefix} file for week {week}")
            paths.extend(found)
        return paths
    return sorted(TRAIN_DIR.glob(f"{prefix}_*.csv"))


def load_raw(weeks: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All requested weeks (config.WEEKS by default) in one combined
    pair of DataFrames. Fine for a handful of weeks; for the full
    18-week set, prefer cache_pipeline.build_dataset_cached, which loads
    (and frees) one week at a time instead of holding every week's raw
    CSV in memory simultaneously -- see TIER1_SPEC_4.md, Task J.
    """
    inp_paths = _week_files("input", weeks)
    out_paths = _week_files("output", weeks)
    if not inp_paths or not out_paths:
        raise FileNotFoundError("no input/output csv pairs found")

    print(f"loading {len(inp_paths)} input and {len(out_paths)} output files")
    inp = pd.concat([pd.read_csv(p, usecols=INPUT_USECOLS) for p in inp_paths],
                    ignore_index=True)
    out = pd.concat([pd.read_csv(p, usecols=OUTPUT_USECOLS) for p in out_paths],
                    ignore_index=True)
    inp = inp.rename(columns=RENAME)
    out = out.rename(columns=RENAME)
    print(f"  input  {len(inp):,} rows")
    print(f"  output {len(out):,} rows")
    return inp, out


def load_raw_week(week: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A single week's input/output, usecols-trimmed. Building block for
    cache_pipeline's one-week-at-a-time loading."""
    return load_raw(weeks=[week])


# ------------------------------------------------------- normalisation

def normalise(inp: pd.DataFrame, out: pd.DataFrame
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror leftward plays so every play runs in +x.

    The output file has no play_direction column of its own, so the flag is
    carried over from the input by play. Forgetting to mirror the output --
    or the ball landing point -- is the single easiest way to produce a
    pipeline that looks fine and scores nonsense.
    """
    play_dir = inp.groupby(KEY)["play_dir"].first().rename("play_dir")

    flip_in = inp["play_dir"].astype(str).str.lower().eq("left").to_numpy()
    out = out.merge(play_dir, on=KEY, how="left")
    flip_out = out["play_dir"].astype(str).str.lower().eq("left").to_numpy()

    inp = inp.copy()
    for df, flip, cols in ((inp, flip_in, ["x", "ball_land_x"]),
                           (out, flip_out, ["x"])):
        for col in cols:
            if col in df.columns:
                df.loc[flip, col] = FIELD_LENGTH - df.loc[flip, col]
    for df, flip, cols in ((inp, flip_in, ["y", "ball_land_y"]),
                           (out, flip_out, ["y"])):
        for col in cols:
            if col in df.columns:
                df.loc[flip, col] = FIELD_WIDTH - df.loc[flip, col]

    for col in ("direction", "orientation"):
        if col in inp.columns:
            inp.loc[flip_in, col] = (inp.loc[flip_in, col] + 180.0) % 360.0

    print(f"  mirrored {flip_in.sum():,} input rows, "
          f"{flip_out.sum():,} output rows")
    return inp, out.drop(columns=["play_dir"])


def add_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """(speed, direction) -> (vx, vy); (accel, direction) -> (ax, ay).

    NGS `dir` is degrees clockwise from the +y axis, not the usual maths
    convention, hence the sin/cos swap.

    `a` is a magnitude with no direction of its own. Projecting it onto the
    current heading assumes all acceleration is along-track: true when a
    player is speeding up in a straight line, false when they cut. That
    assumption is why the constant-acceleration baseline falls apart on
    turns, and it belongs in the write-up rather than hidden here.
    """
    df = df.copy()
    rad = np.deg2rad(df["direction"].to_numpy())
    sin_d, cos_d = np.sin(rad), np.cos(rad)
    df["vx"] = df["speed"].to_numpy() * sin_d
    df["vy"] = df["speed"].to_numpy() * cos_d
    df["ax"] = df["accel"].to_numpy() * sin_d
    df["ay"] = df["accel"].to_numpy() * cos_d
    return df


# ------------------------------------------------------------- samples

def build_samples(inp: pd.DataFrame, out: pd.DataFrame) -> list[PlaySample]:
    """Pair each play's anchor state with its ground-truth future."""
    horizons = (out.groupby(KEY)["frame_id"].max()
                .rename("horizon").reset_index())
    keep = horizons[(horizons["horizon"] >= MIN_HORIZON)
                    & (horizons["horizon"] <= MAX_HORIZON)]
    print(f"  {len(keep):,} of {len(horizons):,} plays have "
          f"{MIN_HORIZON}-{MAX_HORIZON} output frames")

    if MAX_PLAYS and len(keep) > MAX_PLAYS:
        keep = keep.sample(MAX_PLAYS, random_state=0).sort_index()
        print(f"  subsampled to {MAX_PLAYS} plays")

    inp = inp.merge(keep[KEY], on=KEY, how="inner")
    out = out.merge(keep[KEY], on=KEY, how="inner")

    anchor = inp.groupby(KEY)["frame_id"].transform("max")
    at_anchor = inp[inp["frame_id"] == anchor].copy()

    horizon_by_key = {tuple(r[:2]): int(r[2])
                      for r in keep[KEY + ["horizon"]].to_numpy()}

    out = out.sort_values(KEY + ["player_id", "frame_id"])
    context_by_key = dict(tuple(at_anchor.groupby(KEY, sort=False)))
    # `inp` here is already just the pre-throw frames for kept plays -- the
    # input file never contains anything after the throw -- so this is the
    # full history, not a slice of something larger.
    history_by_key = dict(tuple(
        inp[["game_id", "play_id", "player_id", "frame_id", "x", "y",
             "vx", "vy"]].groupby(KEY, sort=False)))

    samples = []
    skipped = 0
    for key, play_out in out.groupby(KEY, sort=False):
        horizon = horizon_by_key[key]
        context = context_by_key.get(key)
        if context is None:
            skipped += 1
            continue

        state = context[context["player_to_predict"]]
        if ROLES_OF_INTEREST:
            state = state[state["role"].isin(ROLES_OF_INTEREST)]
        if state.empty:
            skipped += 1
            continue

        truth = _pivot_truth(play_out, state["player_id"].to_numpy(), horizon)
        if truth is None:
            skipped += 1
            continue

        player_ids = state["player_id"].to_numpy()
        full_history = history_by_key.get(key)
        history = (full_history[full_history["player_id"].isin(player_ids)]
                   .sort_values(["player_id", "frame_id"])
                   .reset_index(drop=True)
                   if full_history is not None
                   else pd.DataFrame(columns=["game_id", "play_id",
                                              "player_id", "frame_id",
                                              "x", "y", "vx", "vy"]))

        samples.append(PlaySample(
            game_id=int(key[0]), play_id=int(key[1]),
            state=state.reset_index(drop=True), history=history, truth=truth,
            horizon=horizon, context=context.reset_index(drop=True)))

    print(f"  built {len(samples):,} samples ({skipped:,} skipped)")
    if not samples:
        raise ValueError("no usable plays -- loosen MIN_HORIZON / MAX_HORIZON")
    return samples


def _pivot_truth(play_out: pd.DataFrame, player_ids: np.ndarray,
                 horizon: int):
    """(n_players, horizon, 2), or None if any player is missing a frame."""
    pivot = play_out.pivot_table(index="player_id", columns="frame_id",
                                 values=["x", "y"])
    try:
        pivot = pivot.loc[player_ids]
    except KeyError:
        return None

    frames = list(range(1, horizon + 1))
    try:
        xs = pivot["x"][frames].to_numpy()
        ys = pivot["y"][frames].to_numpy()
    except KeyError:
        return None
    if np.isnan(xs).any() or np.isnan(ys).any():
        return None
    return np.stack([xs, ys], axis=-1)


# --------------------------------------------------------------- entry

def build_dataset() -> list[PlaySample]:
    inp, out = load_raw()
    inp, out = normalise(inp, out)
    inp = add_kinematics(inp)
    samples = build_samples(inp, out)

    n_players = sum(len(s.state) for s in samples)
    horizons = np.array([s.horizon for s in samples])
    print(f"dataset: {len(samples):,} plays, {n_players:,} player-tracks, "
          f"horizon {horizons.min()}-{horizons.max()} frames "
          f"(median {int(np.median(horizons))} = "
          f"{np.median(horizons) * DT:.1f} s)\n")
    return samples
