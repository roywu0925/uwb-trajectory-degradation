"""Hybrid predictors: toward_ball_land's destination geometry combined
with a Kalman-smoothed position and/or speed.

README.md's Tier 1 Finding 2 showed toward_ball_land decomposes into two
independent pieces: a *direction* (geometric -- anchor position to the
known ball landing point -- and noise-robust, since position noise shifts
the vector's origin without rotating it much) and a *speed magnitude*
(read straight off the single, possibly noisy anchor frame -- not
robust). Plain kalman has the opposite profile: a smoothed speed and no
destination information at all. This module builds the 2x2 of possible
combinations, not just the one hybrid, so a win can be attributed to the
right half:

                          speed: finite difference   speed: Kalman posterior
  anchor: raw measurement   toward_ball_land (baselines.py)   tbl_kalman_speed
  anchor: Kalman posterior  tbl_kalman_pos                     tbl_kalman_both

Direction computation, horizon handling, and bounding behaviour (never
overshoot the landing point) are identical to baselines.toward_ball_land
-- only the anchor-position source and the speed source differ, each
behind its own boolean switch, so the three new predictors are one
function, not three copies.

The Kalman posterior used here is the same position-only filter as
kalman.kalman (H observes x, y only) -- not kalman_velocity -- since the
whole point is a speed that is smoothed from position history rather
than one that has partially trusted a measurement that may itself be
degraded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DT, KALMAN_MIN_HISTORY
from kalman import _run_filter


def _kalman_posterior(grp: pd.DataFrame | None, Q: np.ndarray) -> np.ndarray | None:
    """[x, y, vx, vy] at the anchor from the position-only Kalman filter,
    or None if there's too little history to trust it."""
    if grp is None or len(grp) < KALMAN_MIN_HISTORY:
        return None
    return _run_filter(grp, Q, use_velocity=False)


def make_hybrid(use_kalman_pos: bool, use_kalman_speed: bool, Q: np.ndarray):
    """Return a f(sample, horizon) predictor for one cell of the 2x2.

    `Q` is the frozen Tier 0b/1 process noise (kalman.build_Q(sigma_a)),
    the same value used by every other Kalman-family predictor in the
    sweep -- not fit here. `R` is whatever run_tier1_sweep.set_filter_R
    last set on the kalman module (module-level, picked up automatically
    by kalman._run_filter at call time), so this composes with the
    existing degradation harness without extra wiring.
    """
    short_history_count = 0

    def predict(sample, horizon: int) -> np.ndarray:
        nonlocal short_history_count
        state = sample.state
        p0 = state[["x", "y"]].to_numpy().astype(float).copy()
        speed = state["speed"].to_numpy().astype(float).copy()

        if use_kalman_pos or use_kalman_speed:
            history_by_player = dict(tuple(
                sample.history.groupby("player_id", sort=False)))
            for i, player_id in enumerate(state["player_id"].to_numpy()):
                post = _kalman_posterior(history_by_player.get(player_id), Q)
                if post is None:
                    short_history_count += 1
                    continue
                if use_kalman_pos:
                    p0[i, 0], p0[i, 1] = post[0], post[1]
                if use_kalman_speed:
                    speed[i] = np.hypot(post[2], post[3])

        target = state[["ball_land_x", "ball_land_y"]].to_numpy()
        delta = target - p0
        dist = np.linalg.norm(delta, axis=-1, keepdims=True)
        unit = np.divide(delta, dist, out=np.zeros_like(delta), where=dist > 1e-6)

        t = np.arange(1, horizon + 1) * DT
        reach = speed[:, None] * t                # (n, horizon)
        travel = np.minimum(reach, dist)           # never overshoot, same as toward_ball_land
        return p0[:, None, :] + unit[:, None, :] * travel[:, :, None]

    if use_kalman_pos and use_kalman_speed:
        name = "tbl_kalman_both"
    elif use_kalman_pos:
        name = "tbl_kalman_pos"
    elif use_kalman_speed:
        name = "tbl_kalman_speed"
    else:
        name = "toward_ball_land_hybrid_identity"  # both flags False: unused in practice
    predict.__name__ = name
    predict.short_history_count = lambda: short_history_count
    return predict


def make_hybrid_family(Q: np.ndarray) -> dict:
    """The three new cells of the 2x2 (toward_ball_land itself, the
    fourth cell, already lives in baselines.py)."""
    return {
        "tbl_kalman_speed": make_hybrid(False, True, Q),
        "tbl_kalman_pos": make_hybrid(True, False, Q),
        "tbl_kalman_both": make_hybrid(True, True, Q),
    }
