"""Synthetic sanity checks for kalman.py.

Not a pytest suite -- run directly:

    python test_kalman.py

Two silent bugs have already been caught in this project only by running
code (see CONTEXT.md), and a Kalman filter with a transposed matrix still
produces plausible-looking numbers. This generates motion with a known
answer and checks the filter actually recovers it before it ever touches
real data.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from kalman import (F, build_Q, compute_mean_nis, estimate_sigma_a,
                    make_kalman, _run_filter)

RNG = np.random.default_rng(0)


def _synthetic_history(n_frames: int, x0: float, y0: float,
                       vx: float, vy: float, sigma_pos: float,
                       player_id: int = 1) -> pd.DataFrame:
    """Constant-velocity ground truth + iid Gaussian position noise."""
    dt = 0.1
    t = np.arange(n_frames) * dt
    true_x = x0 + vx * t
    true_y = y0 + vy * t
    meas_x = true_x + RNG.normal(0, sigma_pos, n_frames)
    meas_y = true_y + RNG.normal(0, sigma_pos, n_frames)
    # velocity columns only feed the init average in kalman.py's default
    # (position-only) filter -- give them a bit of noise too so the test
    # doesn't cheat by handing the filter the true velocity for free.
    meas_vx = np.full(n_frames, vx) + RNG.normal(0, 0.3, n_frames)
    meas_vy = np.full(n_frames, vy) + RNG.normal(0, 0.3, n_frames)
    return pd.DataFrame({
        "player_id": player_id,
        "frame_id": np.arange(1, n_frames + 1),
        "x": meas_x, "y": meas_y, "vx": meas_vx, "vy": meas_vy,
    })


def test_recovers_constant_velocity():
    """Filtering many noisy position fixes should land close to the truth,
    much closer than any single measurement -- that's the entire point of
    running a filter instead of just reading the last frame."""
    sigma_pos = 0.25
    n_frames = 40
    vx_true, vy_true = 2.0, -1.0
    x0, y0 = 10.0, 20.0
    grp = _synthetic_history(n_frames, x0, y0, vx_true, vy_true, sigma_pos)

    Q = build_Q(sigma_a=0.5)
    x_est = _run_filter(grp, Q, use_velocity=False)

    true_final = np.array([x0 + vx_true * (n_frames - 1) * 0.1,
                           y0 + vy_true * (n_frames - 1) * 0.1])
    pos_err = np.linalg.norm(x_est[:2] - true_final)
    vel_err = np.linalg.norm(x_est[2:] - [vx_true, vy_true])

    print(f"  position error: {pos_err:.3f} yd (raw measurement noise "
          f"sigma = {sigma_pos:.3f} yd)")
    print(f"  velocity error: {vel_err:.3f} yd/s")

    assert pos_err < sigma_pos, (
        f"filtered position error ({pos_err:.3f}) should beat raw "
        f"measurement noise ({sigma_pos:.3f}) -- filter isn't averaging "
        f"anything, check for a wiring bug")
    assert vel_err < 0.5, (
        f"velocity error {vel_err:.3f} yd/s is too high for {n_frames} "
        f"frames of clean constant-velocity motion")
    print("  PASS")


def test_recovers_stationary_target():
    """vx = vy = 0 is the edge case most likely to break on a sign error
    in F or H (a transposed or flipped matrix can still 'work' for moving
    targets by accident)."""
    sigma_pos = 0.2
    grp = _synthetic_history(30, x0=50.0, y0=25.0, vx=0.0, vy=0.0,
                             sigma_pos=sigma_pos)
    Q = build_Q(sigma_a=0.3)
    x_est = _run_filter(grp, Q, use_velocity=False)

    pos_err = np.linalg.norm(x_est[:2] - [50.0, 25.0])
    vel_err = np.linalg.norm(x_est[2:])
    print(f"  position error: {pos_err:.3f} yd, velocity error: "
          f"{vel_err:.3f} yd/s (both should be small)")
    assert pos_err < sigma_pos
    assert vel_err < 0.3
    print("  PASS")


def test_velocity_variant_uses_measurements():
    """The H=I variant should converge at least as tightly as position-only
    on the same data, since it has strictly more information per step."""
    sigma_pos = 0.25
    grp = _synthetic_history(40, x0=0.0, y0=0.0, vx=3.0, vy=1.5,
                             sigma_pos=sigma_pos)
    Q = build_Q(sigma_a=0.5)
    x_pos_only = _run_filter(grp, Q, use_velocity=False)
    x_pos_vel = _run_filter(grp, Q, use_velocity=True)

    true_vel = np.array([3.0, 1.5])
    err_pos_only = np.linalg.norm(x_pos_only[2:] - true_vel)
    err_pos_vel = np.linalg.norm(x_pos_vel[2:] - true_vel)
    print(f"  velocity error, position-only H: {err_pos_only:.3f} yd/s")
    print(f"  velocity error, position+velocity H: {err_pos_vel:.3f} yd/s")
    assert err_pos_vel < err_pos_only + 0.2, (
        "observing velocity directly should not make the velocity "
        "estimate worse")
    print("  PASS")


def test_sigma_a_estimate_recovers_known_noise():
    """estimate_sigma_a should recover a known injected acceleration
    noise level from synthetic data, up to sampling error."""
    true_sigma_a = 1.5
    n_frames = 200
    dt = 0.1
    vx = np.cumsum(RNG.normal(0, true_sigma_a * dt, n_frames))
    vy = np.cumsum(RNG.normal(0, true_sigma_a * dt, n_frames))
    history = pd.DataFrame({
        "player_id": 1, "frame_id": np.arange(1, n_frames + 1),
        "x": 0.0, "y": 0.0, "vx": vx, "vy": vy,
    })
    sample = SimpleNamespace(history=history)
    est = estimate_sigma_a([sample])
    print(f"  true sigma_a = {true_sigma_a:.3f}, estimated = {est:.3f}")
    assert abs(est - true_sigma_a) < 0.3 * true_sigma_a, (
        f"estimated sigma_a ({est:.3f}) too far from the injected value "
        f"({true_sigma_a:.3f})")
    print("  PASS")


def test_short_history_falls_back_cleanly():
    """make_kalman must not crash or silently produce garbage on a player
    with fewer frames than KALMAN_MIN_HISTORY -- it should fall back to
    constant velocity and say so."""
    from config import KALMAN_MIN_HISTORY
    Q = build_Q(sigma_a=0.5)
    fn = make_kalman(Q, use_velocity=False)

    state = pd.DataFrame({
        "player_id": [1], "x": [10.0], "y": [20.0],
        "vx": [2.0], "vy": [1.0],
    })
    history = _synthetic_history(max(1, KALMAN_MIN_HISTORY - 1),
                                 10.0, 20.0, 2.0, 1.0, sigma_pos=0.2)
    sample = SimpleNamespace(state=state, history=history)

    pred = fn(sample, horizon=5)
    assert pred.shape == (1, 5, 2)
    assert np.isfinite(pred).all()
    assert fn.short_history_count() == 1, "short-history fallback should be counted, not silent"
    print(f"  fell back cleanly, shape {pred.shape}, "
          f"short_history_count={fn.short_history_count()}")
    print("  PASS")


def test_nis_matches_expected_for_well_specified_model():
    """If the synthetic generator matches the filter's own model exactly
    (same sigma_a driving true acceleration, same R corrupting the
    measurements), mean NIS should land close to dim(z). This validates
    the NIS computation itself -- the diagnostic used to check Q against
    real data, where the true generating process is unknown -- before
    trusting its readout there."""
    from config import KALMAN_R_POS, KALMAN_R_VEL

    dt = 0.1
    sigma_a = 1.0
    Q = build_Q(sigma_a)
    n_players, n_frames = 40, 60
    rows = []
    for pid in range(n_players):
        x, y = 0.0, 0.0
        vx, vy = RNG.normal(0, 3), RNG.normal(0, 3)
        for f in range(1, n_frames + 1):
            ax, ay = RNG.normal(0, sigma_a), RNG.normal(0, sigma_a)
            x += vx * dt + 0.5 * ax * dt ** 2
            y += vy * dt + 0.5 * ay * dt ** 2
            vx += ax * dt
            vy += ay * dt
            rows.append((
                pid, f,
                x + RNG.normal(0, np.sqrt(KALMAN_R_POS)),
                y + RNG.normal(0, np.sqrt(KALMAN_R_POS)),
                vx + RNG.normal(0, np.sqrt(KALMAN_R_VEL)),
                vy + RNG.normal(0, np.sqrt(KALMAN_R_VEL)),
            ))
    history = pd.DataFrame(
        rows, columns=["player_id", "frame_id", "x", "y", "vx", "vy"])
    sample = SimpleNamespace(history=history)

    for use_velocity, dim_z in ((False, 2), (True, 4)):
        nis, n = compute_mean_nis([sample], Q, use_velocity)
        print(f"  use_velocity={use_velocity}: mean NIS = {nis:.2f} over "
              f"{n} steps (expected ~= {dim_z})")
        assert abs(nis - dim_z) < 0.3 * dim_z, (
            f"mean NIS {nis:.2f} too far from expected {dim_z} for a "
            f"well-specified synthetic model -- the NIS computation "
            f"itself may be wrong, not just Q")
    print("  PASS")


def run_all():
    tests = [
        test_recovers_constant_velocity,
        test_recovers_stationary_target,
        test_velocity_variant_uses_measurements,
        test_sigma_a_estimate_recovers_known_noise,
        test_short_history_falls_back_cleanly,
        test_nis_matches_expected_for_well_specified_model,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
    print(f"\nall {len(tests)} kalman sanity tests passed")


if __name__ == "__main__":
    run_all()
