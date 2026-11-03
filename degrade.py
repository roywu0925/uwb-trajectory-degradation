"""Sensor degradation for the Tier 1 robustness sweep (TIER1_SPEC.md, Task B).

Corrupts `input_*.csv` positions with Gaussian noise and, for the
"derived" velocity source, re-derives everything downstream from the
corrupted positions the way a real degraded sensor's own pipeline would --
speed and direction included, not just vx/vy. Leaving speed/direction on
the clean NGS values while only touching vx/vy would silently run half
the sweep on uncorrupted data; this project has already lost a day to
exactly that failure shape once (the constant-acceleration baseline
silently degrading to constant-velocity because ax/ay were never
computed -- see CONTEXT.md). `assert_speed_diverged` exists to make that
class of bug loud instead of silent here.

Never touches `output_*.csv` (ground truth) or `ball_land_x`/`ball_land_y`
(a separate, deliberately isolated degradation axis for a later round).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DT

TRACK_KEY = ["game_id", "play_id", "player_id"]


def add_position_noise(df: pd.DataFrame, sigma: float,
                       rng: np.random.Generator) -> pd.DataFrame:
    """Corrupt x, y with iid Gaussian noise, sigma in yards. Nothing else
    is touched here -- ball_land_x/y, speed, direction all pass through
    unchanged at this step."""
    df = df.copy()
    if sigma > 0:
        df["x"] = df["x"].to_numpy() + rng.normal(0, sigma, len(df))
        df["y"] = df["y"].to_numpy() + rng.normal(0, sigma, len(df))
    return df


def derive_kinematics_from_position(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute vx, vy, speed, direction, ax, ay from x, y alone, by
    finite differencing -- the "raw sensor, derive everything downstream"
    path.

    Direction convention: data_loader.add_kinematics maps
    vx = s*sin(dir), vy = s*cos(dir) (NGS `dir` is degrees clockwise from
    +y, not the usual maths convention). The inverse is therefore
    atan2(vx, vy), NOT the usual atan2(vy, vx) -- getting this backwards
    silently rotates every trajectory 90 degrees. Checked against real
    NGS data in smoke_test_degrade.py before trusting it here.

    Acceleration is differenced twice (position -> velocity -> velocity
    residual), so its noise is amplified far more than velocity's --
    `constant_acceleration` should fail catastrophically under this path.
    That is correct behaviour, not a defect (TIER1_SPEC.md).

    Frame 1 of each track has no backward difference to take; its
    velocity/acceleration is back-filled from frame 2's value rather than
    left NaN, so downstream code (state at the anchor, Kalman init) never
    sees an undefined first frame.
    """
    df = df.sort_values(TRACK_KEY + ["frame_id"]).copy()
    grp = df.groupby(TRACK_KEY, sort=False)

    vx = (grp["x"].diff() / DT)
    vy = (grp["y"].diff() / DT)
    vx = vx.groupby([df["game_id"], df["play_id"], df["player_id"]]).bfill()
    vy = vy.groupby([df["game_id"], df["play_id"], df["player_id"]]).bfill()

    df["vx"] = vx.to_numpy()
    df["vy"] = vy.to_numpy()
    df["speed"] = np.hypot(df["vx"].to_numpy(), df["vy"].to_numpy())
    df["direction"] = np.degrees(
        np.arctan2(df["vx"].to_numpy(), df["vy"].to_numpy())) % 360.0

    grp = df.groupby(TRACK_KEY, sort=False)
    ax = (grp["vx"].diff() / DT)
    ay = (grp["vy"].diff() / DT)
    ax = ax.groupby([df["game_id"], df["play_id"], df["player_id"]]).bfill()
    ay = ay.groupby([df["game_id"], df["play_id"], df["player_id"]]).bfill()
    df["ax"] = ax.to_numpy()
    df["ay"] = ay.to_numpy()
    df["accel"] = np.hypot(df["ax"].to_numpy(), df["ay"].to_numpy())

    return df


def assert_speed_diverged(original_speed: pd.Series, derived_speed: pd.Series,
                          sigma: float, tol: float = 0.05) -> None:
    """Fail loudly if degrading positions didn't actually change the
    speed column -- the exact silent-bug shape TIER1_SPEC.md warns
    about (only vx/vy overwritten, speed/direction left on the clean
    NGS values). Indices must align; both Series keep their original
    row labels through add_position_noise/derive_kinematics_from_position
    even though the latter re-sorts row order.
    """
    if sigma <= 0:
        return
    mean_abs_diff = (derived_speed - original_speed).abs().mean()
    assert mean_abs_diff > tol, (
        f"derived speed barely differs from NGS speed (mean |diff| = "
        f"{mean_abs_diff:.4f} yd/s) at sigma={sigma} -- speed/direction "
        f"probably weren't actually overwritten with derived values")


def corrupt_input(inp: pd.DataFrame, sigma: float, rng: np.random.Generator,
                  velocity_source: str) -> pd.DataFrame:
    """Corrupt x, y, then either re-derive kinematics from them
    ("derived") or leave the clean NGS-derived kinematics in place
    ("ngs") -- see config.VELOCITY_SOURCES.

    `inp` must already have gone through data_loader.add_kinematics (so
    "ngs" mode has clean vx/vy/speed/direction/ax/ay to fall back to).
    """
    original_speed = inp["speed"].copy()
    df = add_position_noise(inp, sigma, rng)

    if velocity_source == "derived":
        df = derive_kinematics_from_position(df)
        assert_speed_diverged(original_speed, df["speed"], sigma)
    elif velocity_source != "ngs":
        raise ValueError(f"unknown velocity_source {velocity_source!r}")

    return df


# ---------------------------------------------------- second degradation axis

def add_ball_land_noise(df: pd.DataFrame, sigma_ball: float,
                        rng: np.random.Generator) -> pd.DataFrame:
    """Corrupt ball_land_x/y with iid Gaussian noise, sigma_ball in
    yards (TIER1_SPEC_3.md, Task G).

    Drawn ONCE PER PLAY, not once per frame: the landing point is a
    single estimate for the whole play, not something re-measured every
    frame. Fresh noise per frame would model nothing real -- there is no
    physical process re-estimating the landing point ten times a second
    -- and would average itself away across the play's history frames,
    hiding exactly the systematic bias this axis exists to test.

    Uses whichever `rng` the caller passes, which must be an independent
    stream from the one used for player-position noise
    (add_position_noise) -- never the same generator. That independence
    is what makes sigma_ball=0 an exact, not approximate, no-op: this
    function returns immediately without drawing anything, so it cannot
    perturb the player-noise RNG's sequence even by consuming random
    state. Every Part 2 number must reproduce bit-for-bit at
    sigma_ball=0; if it doesn't, the two noise axes are coupled and nothing
    downstream can be trusted (see run_tier1_twoaxis.py's regression check).
    """
    df = df.copy()
    if sigma_ball <= 0:
        return df

    keys = df[["game_id", "play_id"]].drop_duplicates().reset_index(drop=True)
    keys["_dx"] = rng.normal(0, sigma_ball, len(keys))
    keys["_dy"] = rng.normal(0, sigma_ball, len(keys))

    df = df.merge(keys, on=["game_id", "play_id"], how="left")
    df["ball_land_x"] = df["ball_land_x"] + df["_dx"]
    df["ball_land_y"] = df["ball_land_y"] + df["_dy"]
    return df.drop(columns=["_dx", "_dy"])


def corrupt_input_two_axis(inp: pd.DataFrame, sigma_player: float,
                          sigma_ball: float, player_rng: np.random.Generator,
                          ball_rng: np.random.Generator,
                          velocity_source: str) -> pd.DataFrame:
    """corrupt_input (player-position axis) followed by
    add_ball_land_noise (landing-point axis), each with its own RNG
    stream. The two must never share a generator -- see
    add_ball_land_noise's docstring for why.
    """
    df = corrupt_input(inp, sigma_player, player_rng, velocity_source)
    return add_ball_land_noise(df, sigma_ball, ball_rng)
