"""Synthetic sanity checks for imm.py.

Run directly:

    python test_imm.py

The critical failure mode for an IMM is a broken mixing step that still
produces plausible-looking position output -- two filters idling in
parallel rather than interacting. Position error alone won't catch that;
the model probabilities will. This generates a trajectory with a known
regime change and checks the filter actually detects it.

Note on test parameters: these use sigma_steady = 0.3, not the real
NIS-calibrated 1.791 the actual pipeline uses (see run_tier0a.py). A
first pass at this test with sigma_steady = 1.791 and a modest jerk
showed almost no discrimination between the two models -- the steady
model's own per-step process-noise budget (std ~= 1.8 yd/s^2 per frame)
already tolerates the sort of acceleration change that was meant to
represent a "manoeuvre", so nothing looked surprising to it. That's a
real, useful finding about the actual filter (see CONTEXT.md), but it
makes a bad unit test: it can't tell "the mixing math is broken" apart
from "the calibrated Q happens to make both models fit similarly well
at these frame rates". sigma_steady = 0.3 here gives a model pair with a
clean separation, so this test isolates the mixing/combination logic,
not the calibrated constants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import IMM_MANOEUVRE_RATIO, IMM_TRANSITION
from imm import _run_imm, build_Q

RNG = np.random.default_rng(0)


def _synthetic_trajectory(n_frames: int, switch: int, jerk: float,
                          sigma_pos: float, sigma_vel: float,
                          player_id: int = 1) -> pd.DataFrame:
    """Constant velocity up to `switch`, then a sustained acceleration
    ramp (constant nonzero jerk) after it.

    A single constant-acceleration *step* is a bad test case here: once
    the filter's acceleration state catches up to the new constant value
    (a couple of frames), predicting one step further needs no more
    process noise than the steady model already budgets for -- constant
    acceleration has zero jerk, so it doesn't sustain a "manoeuvring"
    verdict past the initial transient. A ramp keeps producing nonzero
    jerk for the whole second half, which is what actually needs the
    manoeuvring model's larger Q throughout.
    """
    dt = 0.1
    x, y, vx, vy, ax = 0.0, 0.0, 3.0, 0.0, 0.0
    xs, ys, vxs, vys = [], [], [], []
    for f in range(n_frames):
        if f >= switch:
            ax += jerk * dt
        vx += ax * dt
        x += vx * dt
        y += vy * dt
        xs.append(x); ys.append(y); vxs.append(vx); vys.append(vy)
    return pd.DataFrame({
        "player_id": player_id,
        "frame_id": np.arange(1, n_frames + 1),
        "x": np.array(xs) + RNG.normal(0, sigma_pos, n_frames),
        "y": np.array(ys) + RNG.normal(0, sigma_pos, n_frames),
        "vx": np.array(vxs) + RNG.normal(0, sigma_vel, n_frames),
        "vy": np.array(vys) + RNG.normal(0, sigma_vel, n_frames),
    })


def test_imm_switches_to_manoeuvring_after_a_break():
    """The required test: a receiver who runs straight and then breaks
    should pull the IMM's model probability toward 'manoeuvring' after
    the break, and it should still favour 'steady' before it."""
    n_frames, switch = 60, 30
    sigma_steady = 0.3
    grp = _synthetic_trajectory(n_frames, switch, jerk=40.0,
                                sigma_pos=0.25, sigma_vel=0.5)

    Q_steady = build_Q(sigma_steady)
    Q_manoeuvre = build_Q(sigma_steady * IMM_MANOEUVRE_RATIO)
    sink = []
    _run_imm(grp, Q_steady, Q_manoeuvre, IMM_TRANSITION, mu_sink=sink)

    frame_id = np.array([r[0] for r in sink])
    mu_manoeuvre = np.array([r[2] for r in sink])

    before = mu_manoeuvre[frame_id <= switch].mean()
    # give the sticky transition matrix a few frames to react
    after = mu_manoeuvre[frame_id > switch + 5].mean()
    print(f"  mean P(manoeuvring) before the break: {before:.3f}")
    print(f"  mean P(manoeuvring) well after the break: {after:.3f}")

    assert before < 0.3, (
        f"P(manoeuvring) = {before:.3f} during genuinely constant-"
        f"velocity motion is too high -- check the steady model's Q")
    assert after > before + 0.3, (
        "model probability should shift meaningfully toward manoeuvring "
        "after a sustained real acceleration -- if this fails, the "
        "mixing step is likely broken (two filters idling in parallel, "
        "not interacting)")
    assert after > 0.5, "should favour manoeuvring once clearly accelerating"
    print("  PASS")


def test_imm_stays_steady_without_a_break():
    """No regime change anywhere -- P(manoeuvring) should stay low
    throughout, not drift or oscillate on its own."""
    grp = _synthetic_trajectory(60, switch=10_000, jerk=0.0,
                                sigma_pos=0.25, sigma_vel=0.5)
    sigma_steady = 0.3
    Q_steady = build_Q(sigma_steady)
    Q_manoeuvre = build_Q(sigma_steady * IMM_MANOEUVRE_RATIO)
    sink = []
    _run_imm(grp, Q_steady, Q_manoeuvre, IMM_TRANSITION, mu_sink=sink)

    mu_manoeuvre = np.array([r[2] for r in sink])
    late = mu_manoeuvre[-20:].mean()
    print(f"  mean P(manoeuvring) late in a pure constant-velocity run: "
          f"{late:.3f}")
    assert late < 0.3, "should not drift toward manoeuvring without cause"
    print("  PASS")


def test_imm_recovers_constant_velocity_position():
    """Sanity check on the combined output, not just the probabilities:
    with no regime change, the combined state estimate shouldn't diverge
    from the (noisy) measurements it was given."""
    n_frames = 40
    sigma_pos = 0.25
    grp = _synthetic_trajectory(n_frames, switch=10_000, jerk=0.0,
                                sigma_pos=sigma_pos, sigma_vel=0.5)
    Q_steady = build_Q(0.3)
    Q_manoeuvre = build_Q(0.3 * IMM_MANOEUVRE_RATIO)
    x_est, mu = _run_imm(grp, Q_steady, Q_manoeuvre, IMM_TRANSITION)

    true_final = np.array([grp["x"].iloc[-1], grp["y"].iloc[-1]])
    pos_err = np.linalg.norm(x_est[:2] - true_final)
    print(f"  position error vs last (noisy) measurement: {pos_err:.3f} yd")
    assert pos_err < 3 * sigma_pos, "combined estimate diverged"
    print("  PASS")


def run_all():
    tests = [
        test_imm_switches_to_manoeuvring_after_a_break,
        test_imm_stays_steady_without_a_break,
        test_imm_recovers_constant_velocity_position,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
    print(f"\nall {len(tests)} imm sanity tests passed")


if __name__ == "__main__":
    run_all()
