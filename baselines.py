"""Physics-only predictors. No learning, no fitting.

Signature, shared by every predictor so they stay interchangeable:

    f(sample, horizon) -> ndarray (n_players, horizon, 2)

`sample` is a data_loader.PlaySample. `sample.state` is one row per player
at the anchor frame, with at least x, y, vx, vy, ax, ay, speed,
ball_land_x, ball_land_y. Predictors that only need the anchor snapshot
take `sample.state` and ignore the rest; Kalman is the first one that also
needs `sample.history`, which is why the signature carries the whole
sample rather than just the state frame.

These are the floor. Whatever gets built later has to beat them, and the
size of the gap -- not the absolute error -- is the result worth reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DT


def _blank(n: int, horizon: int) -> np.ndarray:
    return np.zeros((n, horizon, 2), dtype=float)


def _t(horizon: int) -> np.ndarray:
    return np.arange(1, horizon + 1) * DT


def constant_position(sample, horizon: int) -> np.ndarray:
    """Nobody moves. The absolute floor.

    Keep it: if a sophisticated model ever loses to this, something is
    wired backwards, and that is cheap to discover here.
    """
    state = sample.state
    out = _blank(len(state), horizon)
    out[:, :, 0] = state["x"].to_numpy()[:, None]
    out[:, :, 1] = state["y"].to_numpy()[:, None]
    return out


def constant_velocity(sample, horizon: int) -> np.ndarray:
    """p(t) = p0 + v*t. Five lines, and hard to beat in the first second."""
    state = sample.state
    t = _t(horizon)
    out = _blank(len(state), horizon)
    out[:, :, 0] = state["x"].to_numpy()[:, None] + state["vx"].to_numpy()[:, None] * t
    out[:, :, 1] = state["y"].to_numpy()[:, None] + state["vy"].to_numpy()[:, None] * t
    return out


def constant_acceleration(sample, horizon: int) -> np.ndarray:
    """p(t) = p0 + v*t + a*t^2/2.

    Expected to lose to constant velocity past roughly one second: nobody
    sustains their instantaneous acceleration for three, so the quadratic
    term runs away. That divergence is a finding, not a bug -- it is the
    first place a purely kinematic description visibly stops describing
    people.
    """
    state = sample.state
    t = _t(horizon)
    half_t2 = 0.5 * t ** 2
    out = _blank(len(state), horizon)
    out[:, :, 0] = (state["x"].to_numpy()[:, None]
                    + state["vx"].to_numpy()[:, None] * t
                    + state["ax"].to_numpy()[:, None] * half_t2)
    out[:, :, 1] = (state["y"].to_numpy()[:, None]
                    + state["vy"].to_numpy()[:, None] * t
                    + state["ay"].to_numpy()[:, None] * half_t2)
    return out


def damped_velocity(sample, horizon: int, tau: float = 1.2) -> np.ndarray:
    """Velocity decaying with time constant tau.

    One parameter admitting that players slow down and change direction.
    Usually the strongest purely kinematic baseline.
    """
    state = sample.state
    travel = tau * (1.0 - np.exp(-_t(horizon) / tau))
    out = _blank(len(state), horizon)
    out[:, :, 0] = state["x"].to_numpy()[:, None] + state["vx"].to_numpy()[:, None] * travel
    out[:, :, 1] = state["y"].to_numpy()[:, None] + state["vy"].to_numpy()[:, None] * travel
    return out


def toward_ball_land(sample, horizon: int) -> np.ndarray:
    """Run straight at the ball's landing spot, at the current speed.

    The landing point is given at prediction time, which is unusual and
    strong: it encodes where the play is going. This baseline ignores
    current heading entirely and uses only that destination, so comparing
    it against constant_velocity separates "where they are going" from
    "where they are pointed".

    Note for the write-up: in a live system the landing point would itself
    be an estimate from the tracking data, carrying its own error. Treating
    it as exact here is a modelling choice, and injecting error into it is
    a natural extra axis for the Tier 1 degradation study.
    """
    state = sample.state
    p0 = state[["x", "y"]].to_numpy()
    target = state[["ball_land_x", "ball_land_y"]].to_numpy()
    delta = target - p0
    dist = np.linalg.norm(delta, axis=-1, keepdims=True)
    unit = np.divide(delta, dist, out=np.zeros_like(delta), where=dist > 1e-6)

    reach = state["speed"].to_numpy()[:, None] * _t(horizon)   # (n, horizon)
    travel = np.minimum(reach, dist)                           # never overshoot
    return p0[:, None, :] + unit[:, None, :] * travel[:, :, None]


def velocity_turning_to_ball(sample, horizon: int,
                             tau: float = 0.8) -> np.ndarray:
    """Start on the current heading, turn towards the landing spot.

    A cheap stand-in for intent: momentum dominates early, destination
    dominates late. If a learned model cannot beat this, it has not
    learned anything the geometry did not already say.
    """
    cv = constant_velocity(sample, horizon)
    ball = toward_ball_land(sample, horizon)
    w = (1.0 - np.exp(-_t(horizon) / tau))[None, :, None]
    return (1.0 - w) * cv + w * ball


BASELINES = {
    "constant_position": constant_position,
    "constant_velocity": constant_velocity,
    "constant_acceleration": constant_acceleration,
    "damped_velocity": damped_velocity,
    "toward_ball_land": toward_ball_land,
    "velocity_turning_to_ball": velocity_turning_to_ball,
}
