"""Kalman-filter baseline: state estimation over the pre-throw history.

Same predictor signature as everything in baselines.py --
`f(sample, horizon) -> ndarray (n_players, horizon, 2)` -- but this is the
first predictor that reads `sample.history` instead of just `sample.state`.

State is `[x, y, vx, vy]`. For every player being predicted, the filter
runs predict/update forward over that player's pre-throw frames (position
measurements only, by default), then extrapolates the posterior at the
anchor forward `horizon` steps at constant velocity -- there are no
measurements after the throw, so those steps are predict-only.

No filterpy: this is hand-written numpy so every line is explainable,
per TIER0B_SPEC.md.

No leakage: `Q` is estimated from `sample.history` (input-only) across the
whole dataset, once, before any play is scored. `R` is a config constant
sourced from the UWB spec, not fitted. Neither ever touches `output_*.csv`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DT, KALMAN_MIN_HISTORY, KALMAN_R_POS, KALMAN_R_VEL

F = np.array([[1.0, 0.0, DT, 0.0],
              [0.0, 1.0, 0.0, DT],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])

H_POS = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]])
H_POS_VEL = np.eye(4)

R_POS = np.diag([KALMAN_R_POS, KALMAN_R_POS])
R_POS_VEL = np.diag([KALMAN_R_POS, KALMAN_R_POS, KALMAN_R_VEL, KALMAN_R_VEL])

# Loose initial uncertainty: position is a real (noisy) measurement at
# frame 1, velocity is an average over a handful of frames and much less
# trustworthy. Not tuned -- just "big enough that the first few updates
# dominate the init", which the filter converges past within a few frames
# regardless of the exact value.
P_INIT = np.diag([1.0, 1.0, 4.0, 4.0])


def build_Q(sigma_a: float) -> np.ndarray:
    """Standard discretised constant-velocity process noise.

    Both blocks come from the same unmodelled acceleration over one step,
    so the position and velocity blocks are correlated -- this is the
    textbook white-noise-acceleration Q, not an ad hoc diagonal one.
    """
    dt = DT
    return sigma_a ** 2 * np.array([
        [dt**4 / 4, 0.0,        dt**3 / 2, 0.0       ],
        [0.0,        dt**4 / 4, 0.0,        dt**3 / 2],
        [dt**3 / 2, 0.0,        dt**2,      0.0       ],
        [0.0,        dt**3 / 2, 0.0,        dt**2     ],
    ])


def estimate_sigma_a(samples: list) -> float:
    """sigma_a from implied acceleration in the input history.

    implied_a = (v[t+1] - v[t]) / dt, pooled over every player, every
    step, both axes. This over-estimates the true process-noise sigma_a
    for two reasons that pull the same way: differencing NGS's own noisy
    velocity amplifies that noise, and real players genuinely accelerate
    and cut -- motion a constant-velocity model treats as "process
    noise" even though it is not noise at all.

    Diagnostic only. `calibrate_sigma_a` picks the value actually used,
    by NIS consistency rather than by this estimator -- and the fact
    that the NIS-consistent value scores *worse* on held-out extrapolation
    than this raw one (see CONTEXT.md) is itself informative: it means
    the second reason dominates the first. A filter tuned to be
    "internally consistent" ends up trusting its own constant-velocity
    prediction more, which is exactly wrong when players are cutting.
    """
    diffs = []
    for sample in samples:
        for _, grp in sample.history.groupby("player_id"):
            grp = grp.sort_values("frame_id")
            vx = grp["vx"].to_numpy()
            vy = grp["vy"].to_numpy()
            if len(vx) < 2:
                continue
            diffs.append(np.diff(vx) / DT)
            diffs.append(np.diff(vy) / DT)
    implied_a = np.concatenate(diffs)
    return float(implied_a.std())


def _init_state(grp: pd.DataFrame) -> np.ndarray:
    x0, y0 = grp["x"].iloc[0], grp["y"].iloc[0]
    n_init = min(KALMAN_MIN_HISTORY, len(grp))
    vx0 = grp["vx"].iloc[:n_init].mean()
    vy0 = grp["vy"].iloc[:n_init].mean()
    return np.array([x0, y0, vx0, vy0])


def _run_filter(grp: pd.DataFrame, Q: np.ndarray, use_velocity: bool,
                nis_sink: list | None = None) -> np.ndarray:
    """Predict/update over one player's history; returns the final state.

    If `nis_sink` is given, appends the normalised innovation squared
    (z - H x-)^T (H P- H^T + R)^-1 (z - H x-) at every update step -- the
    prior x-, P- and the measurement z actually used, nothing else. Used
    for the NIS consistency check, never for scoring against truth.
    """
    x = _init_state(grp)
    P = P_INIT.copy()
    H = H_POS_VEL if use_velocity else H_POS
    R = R_POS_VEL if use_velocity else R_POS

    grp = grp.sort_values("frame_id")
    for i, row in enumerate(grp.itertuples()):
        if i == 0:
            continue  # frame 1 already used to initialise x
        x = F @ x
        P = F @ P @ F.T + Q

        z = (np.array([row.x, row.y, row.vx, row.vy]) if use_velocity
            else np.array([row.x, row.y]))
        innovation = z - H @ x
        S = H @ P @ H.T + R
        if nis_sink is not None:
            nis_sink.append(float(innovation @ np.linalg.inv(S) @ innovation))
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ innovation
        P = P - K @ H @ P
    return x


def run_filter_sequence(grp: pd.DataFrame, Q: np.ndarray,
                        use_velocity: bool = False) -> np.ndarray:
    """Same predict/update loop as `_run_filter`, but returns the filtered
    state at every history frame (L, 4) instead of only the final one --
    causal at every step (frame t's state uses only frames <=t), no
    lookahead. Used by model.py's K4 feature path to replace the raw
    finite-differenced velocity channel with the position-only Kalman
    filter's own velocity estimate, frame by frame -- the same
    tbl_kalman_speed mechanism (TIER1_SPEC_6.md's Task M follow-up), just
    applied at every history step rather than only at the anchor.
    """
    x = _init_state(grp)
    P = P_INIT.copy()
    H = H_POS_VEL if use_velocity else H_POS
    R = R_POS_VEL if use_velocity else R_POS

    grp = grp.sort_values("frame_id")
    states = [x.copy()]
    for i, row in enumerate(grp.itertuples()):
        if i == 0:
            continue
        x = F @ x
        P = F @ P @ F.T + Q
        z = (np.array([row.x, row.y, row.vx, row.vy]) if use_velocity
            else np.array([row.x, row.y]))
        innovation = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ innovation
        P = P - K @ H @ P
        states.append(x.copy())
    return np.array(states)


def calibrate_sigma_a(samples: list, sigma_a_hi: float, target: float = 2.0,
                      tol: float = 0.02, max_iter: int = 40) -> float:
    """Bisect sigma_a so mean NIS (position-only H) lands at `target`.

    Answer-blind: NIS only uses the filter's own priors and the
    measurements it consumes, never output_*.csv. Increasing sigma_a
    grows Q, which grows P- and therefore S = H P- H^T + R, which shrinks
    NIS -- so NIS is monotonically decreasing in sigma_a, and bisection
    applies directly. `sigma_a_hi` should be a value already known to
    give NIS below target (e.g. the raw differencing estimate); the
    search starts from there down to a small positive value.

    Calibrated only against the position-only channel, where R_POS is a
    sourced config constant -- not against the velocity variant, whose
    R_VEL is a placeholder (see config.py). Q describes the physical
    process and is shared by both variants; only the velocity variant's
    R is separately suspect, and that is reported, not corrected here.
    """
    lo, hi = 1e-3, sigma_a_hi
    nis_hi, _ = compute_mean_nis(samples, build_Q(hi), use_velocity=False)
    nis_lo, _ = compute_mean_nis(samples, build_Q(lo), use_velocity=False)
    if not (nis_lo > target > nis_hi):
        raise ValueError(
            f"NIS not bracketed: NIS({lo=}) = {nis_lo:.2f}, "
            f"NIS({hi=}) = {nis_hi:.2f}, target = {target} -- "
            f"widen the search range instead of trusting a bad bisection")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        nis_mid, _ = compute_mean_nis(samples, build_Q(mid), use_velocity=False)
        if abs(nis_mid - target) < tol:
            return mid
        if nis_mid > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compute_mean_nis(samples: list, Q: np.ndarray, use_velocity: bool
                     ) -> tuple[float, int]:
    """Average NIS over every update step, every player, every play.

    Answer-blind: uses only each step's prior (x-, P-, built from Q, R
    and the history so far) and that step's measurement -- never
    output_*.csv. Expected value is dim(z) (2 position-only, 4 with
    velocity) if Q and R are well specified for this data; well below
    that means the filter is carrying more uncertainty than it needs
    (Q or R too large), well above means too little (Q or R too small).
    """
    nis_values: list = []
    for sample in samples:
        for _, grp in sample.history.groupby("player_id"):
            if len(grp) < KALMAN_MIN_HISTORY:
                continue
            _run_filter(grp, Q, use_velocity, nis_sink=nis_values)
    return float(np.mean(nis_values)), len(nis_values)


def _extrapolate(x: np.ndarray, horizon: int) -> np.ndarray:
    t = np.arange(1, horizon + 1) * DT
    px = x[0] + x[2] * t
    py = x[1] + x[3] * t
    return np.stack([px, py], axis=-1)


def make_kalman(Q: np.ndarray, use_velocity: bool = False):
    """Return a f(sample, horizon) predictor closed over a fixed Q."""
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
                # Too little history to trust the filter's convergence --
                # fall back to plain constant velocity from the anchor
                # state rather than running predict/update on noise.
                short_history_count += 1
                x = np.array([row.x, row.y, row.vx, row.vy])
            else:
                x = _run_filter(grp, Q, use_velocity)
            out[i] = _extrapolate(x, horizon)
        return out

    predict.__name__ = "kalman_velocity" if use_velocity else "kalman"
    predict.short_history_count = lambda: short_history_count
    return predict
