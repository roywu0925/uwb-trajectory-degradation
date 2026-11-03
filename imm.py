"""Interacting Multiple Model (IMM) filter: two motion regimes, not one.

The single-model Kalman filter in kalman.py can only be "more responsive"
(large Q) or "more smoothed" (small Q) uniformly across the whole play. The
NIS diagnostic showed that trade-off can't be won with one Q: players are
genuinely constant-velocity most of the time and then genuinely
accelerate/cut, and no single process-noise magnitude describes both. IMM
runs two filters with different Q, in parallel, and lets the data pick
which one to believe at each frame.

Same predictor signature as everything else --
`f(sample, horizon) -> ndarray (n_players, horizon, 2)`.

State is `[x, y, vx, vy, ax, ay]`, 6-dim, shared by both models. Canonically
a "steady" model would be 4-dim (constant velocity) and a "manoeuvring"
model would be 6-dim (constant acceleration), and the IMM mixing step would
need to pad or truncate the state vector between them every step -- that
is the fiddliest part of a textbook IMM and the easiest place to hide a
bug. Sharing one 6-dim state for both models sidesteps that: the two
models then differ *only* in how much process noise they inject into the
acceleration terms, via Q. This is a deliberate simplification, not an
oversight.

H observes `[x, y, vx, vy]` -- the same configuration as
kalman.kalman_velocity, which beat position-only at every horizon.

No IMM library: ~80 lines of numpy, hand-written for the same reason as
kalman.py -- the point of Tier 0b is being able to explain every line.

No leakage: Q_steady's scale comes from kalman.calibrate_sigma_a (answer-
blind, computed once per dataset). The manoeuvring/steady ratio and the
model-transition matrix are fixed design choices in config.py, stated as
such and never adjusted against the error tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (DT, IMM_MANOEUVRE_RATIO, IMM_TRANSITION,
                    KALMAN_MIN_HISTORY, KALMAN_R_POS, KALMAN_R_VEL)
from kalman import _extrapolate

F = np.array([
    [1.0, 0.0, DT, 0.0, 0.5 * DT**2, 0.0         ],
    [0.0, 1.0, 0.0, DT, 0.0,         0.5 * DT**2],
    [0.0, 0.0, 1.0, 0.0, DT,         0.0         ],
    [0.0, 0.0, 0.0, 1.0, 0.0,        DT          ],
    [0.0, 0.0, 0.0, 0.0, 1.0,        0.0         ],
    [0.0, 0.0, 0.0, 0.0, 0.0,        1.0         ],
])

H = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
])

R = np.diag([KALMAN_R_POS, KALMAN_R_POS, KALMAN_R_VEL, KALMAN_R_VEL])

# Position/velocity blocks as loose as the single-model filter's P_INIT;
# acceleration starts at 0 with a wide prior since nothing has observed it
# yet -- not tuned, just "big enough to not bias the first few updates".
P_INIT = np.diag([1.0, 1.0, 4.0, 4.0, 9.0, 9.0])


def build_Q(sigma_a: float) -> np.ndarray:
    """Discretised Wiener-process-acceleration noise for the 6-dim state.

    One order higher than kalman.build_Q's white-noise-acceleration model:
    here acceleration is itself a filtered state, driven by white noise
    with spectral density sigma_a**2, integrated through jerk -> accel ->
    vel -> pos via the noise gain g = [dt^2/2, dt, 1] per axis. The x and
    y axes are independent, so the two 3x3 blocks (on indices [x,vx,ax]
    and [y,vy,ay]) sit block-diagonal in an otherwise-zero 6x6 matrix.
    """
    g = np.array([0.5 * DT**2, DT, 1.0])
    block = sigma_a ** 2 * np.outer(g, g)
    Q = np.zeros((6, 6))
    for idx in ([0, 2, 4], [1, 3, 5]):
        Q[np.ix_(idx, idx)] = block
    return Q


def _init_state(grp: pd.DataFrame) -> np.ndarray:
    x0, y0 = grp["x"].iloc[0], grp["y"].iloc[0]
    n_init = min(KALMAN_MIN_HISTORY, len(grp))
    vx0 = grp["vx"].iloc[:n_init].mean()
    vy0 = grp["vy"].iloc[:n_init].mean()
    return np.array([x0, y0, vx0, vy0, 0.0, 0.0])


def _log_gaussian_likelihood(innovation: np.ndarray, S: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(S)
    dim = innovation.shape[0]
    return -0.5 * (innovation @ np.linalg.solve(S, innovation)
                  + logdet + dim * np.log(2 * np.pi))


def _run_imm(grp: pd.DataFrame, Q_steady: np.ndarray, Q_manoeuvre: np.ndarray,
            transition: np.ndarray, mu_sink: list | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
    """The three IMM steps -- mixing, filtering, combination -- per frame.

    Returns (combined_final_state, final_model_probabilities). If
    `mu_sink` is a list, appends (frame_id, mu_steady, mu_manoeuvre) for
    every frame processed, including the init frame.
    """
    grp = grp.sort_values("frame_id")
    Qs = [Q_steady, Q_manoeuvre]
    x0 = _init_state(grp)
    x = [x0.copy(), x0.copy()]
    P = [P_INIT.copy(), P_INIT.copy()]
    mu = np.array([0.5, 0.5])

    rows = list(grp.itertuples())
    if mu_sink is not None:
        mu_sink.append((rows[0].frame_id, mu[0], mu[1]))

    for row in rows[1:]:
        # 1. Mixing: probability that the previous model was i, given the
        # target model is j -- this is what makes IMM "interacting" rather
        # than two filters running side by side.
        c = transition.T @ mu                                # c[j]
        mu_ij = (transition * mu[:, None]) / c[None, :]       # mu_ij[i, j]

        x0j = [sum(mu_ij[i, j] * x[i] for i in range(2)) for j in range(2)]
        P0j = []
        for j in range(2):
            acc = np.zeros((6, 6))
            for i in range(2):
                d = (x[i] - x0j[j]).reshape(-1, 1)
                acc += mu_ij[i, j] * (P[i] + d @ d.T)
            P0j.append(acc)

        # 2. Filtering: each model predicts/updates independently from its
        # own mixed prior.
        z = np.array([row.x, row.y, row.vx, row.vy])
        new_x, new_P, log_lik = [], [], []
        for j in range(2):
            x_pred = F @ x0j[j]
            P_pred = F @ P0j[j] @ F.T + Qs[j]
            innovation = z - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            new_x.append(x_pred + K @ innovation)
            new_P.append(P_pred - K @ H @ P_pred)
            log_lik.append(_log_gaussian_likelihood(innovation, S))

        # 3. Combination: update model probabilities from each model's
        # likelihood, weighted by the mixing normaliser c.
        log_num = np.array([log_lik[j] + np.log(c[j]) for j in range(2)])
        log_num -= log_num.max()          # numerically stable softmax
        w = np.exp(log_num)
        mu = w / w.sum()

        x, P = new_x, new_P
        if mu_sink is not None:
            mu_sink.append((row.frame_id, mu[0], mu[1]))

    combined_x = mu[0] * x[0] + mu[1] * x[1]
    return combined_x, mu


def _run_single(grp: pd.DataFrame, Q: np.ndarray) -> np.ndarray:
    """One 6-dim constant-acceleration filter, no mixing -- the IMM's
    'steady' model run in isolation, to separate its two possible sources
    of improvement over kalman.kalman_velocity: a richer state (explicit
    acceleration) versus actual regime switching between two models.
    """
    grp = grp.sort_values("frame_id")
    x = _init_state(grp)
    P = P_INIT.copy()
    for i, row in enumerate(grp.itertuples()):
        if i == 0:
            continue
        x = F @ x
        P = F @ P @ F.T + Q
        z = np.array([row.x, row.y, row.vx, row.vy])
        innovation = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ innovation
        P = P - K @ H @ P
    return x


def make_kalman_ca(sigma_a: float):
    """Standalone 6-dim constant-acceleration Kalman filter -- no IMM, no
    mixing, no model probabilities. Same H, R and process noise as the
    IMM's steady model alone. If this matches `imm`'s accuracy, the gain
    IMM shows over kalman.kalman_velocity is coming from the richer
    6-dim state (explicit, lightly-regularised acceleration), not from
    switching between models -- see CONTEXT.md, Experiment A.
    """
    Q = build_Q(sigma_a)
    short_history_count = 0

    def predict(sample, horizon: int) -> np.ndarray:
        nonlocal short_history_count
        state = sample.state
        history_by_player = dict(tuple(
            sample.history.groupby("player_id", sort=False)))

        out = np.zeros((len(state), horizon, 2))
        for i, row in enumerate(state.itertuples()):
            grp = history_by_player.get(row.player_id)
            if grp is None or len(grp) < KALMAN_MIN_HISTORY:
                short_history_count += 1
                x = np.array([row.x, row.y, row.vx, row.vy, 0.0, 0.0])
            else:
                x = _run_single(grp, Q)
            out[i] = _extrapolate(x, horizon)
        return out

    predict.__name__ = "kalman_ca"
    predict.short_history_count = lambda: short_history_count
    return predict


def make_imm(sigma_steady: float, mu_log: list | None = None):
    """Return a f(sample, horizon) predictor using the 2-model IMM.

    `sigma_steady` should be the NIS-calibrated sigma_a from
    kalman.calibrate_sigma_a; the manoeuvring model's sigma_a is
    IMM_MANOEUVRE_RATIO times larger (config.py, a fixed design choice).

    If `mu_log` is given, every scored player-track appends its full
    model-probability trace to it as
    (game_id, play_id, player_id, frame_id, mu_steady, mu_manoeuvre) rows
    -- meant to be dumped to a DataFrame/CSV after evaluate() runs.
    """
    Q_steady = build_Q(sigma_steady)
    Q_manoeuvre = build_Q(sigma_steady * IMM_MANOEUVRE_RATIO)
    short_history_count = 0

    def predict(sample, horizon: int) -> np.ndarray:
        nonlocal short_history_count
        state = sample.state
        history_by_player = dict(tuple(
            sample.history.groupby("player_id", sort=False)))

        out = np.zeros((len(state), horizon, 2))
        for i, row in enumerate(state.itertuples()):
            grp = history_by_player.get(row.player_id)
            if grp is None or len(grp) < KALMAN_MIN_HISTORY:
                short_history_count += 1
                x = np.array([row.x, row.y, row.vx, row.vy, 0.0, 0.0])
            else:
                sink = [] if mu_log is not None else None
                x, _ = _run_imm(grp, Q_steady, Q_manoeuvre, IMM_TRANSITION,
                                mu_sink=sink)
                if mu_log is not None:
                    for frame_id, mu_s, mu_m in sink:
                        mu_log.append((sample.game_id, sample.play_id,
                                       row.player_id, frame_id, mu_s, mu_m))
            out[i] = _extrapolate(x, horizon)
        return out

    predict.__name__ = "imm"
    predict.short_history_count = lambda: short_history_count
    return predict
