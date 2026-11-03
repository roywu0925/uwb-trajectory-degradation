"""Smoke tests for degrade.py, run before the full Tier 1 Task B sweep.

    python smoke_test_degrade.py

Three checks, in the order TIER1_SPEC.md asks for them:

1. The silent-bug guard (assert_speed_diverged) actually fires on the
   exact failure shape it exists to catch -- vx/vy overwritten from
   corrupted positions, speed/direction left on the clean NGS values --
   and does not fire on the correct path.
2. The atan2 direction convention round-trips against real NGS data:
   derive velocity from clean (uncorrupted) positions by finite
   differencing, convert back to a direction, and compare against NGS's
   own `direction` column on frames where a player is moving in a
   straight line. Agreement should be close (a few degrees); 90 degrees
   off means the atan2 argument order is backwards.
3. A tiny end-to-end run of the real sweep machinery (MAX_PLAYS=50, one
   sigma) to catch any wiring problem before spending time on the full
   36-combination sweep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import degrade
from config import DEGRADE_BASE_SEED, DT
from data_loader import add_kinematics, load_raw, normalise
from run_tier1_sweep import frozen_sigma_a, run_sweep


def _broken_derive_missing_overwrite(df: pd.DataFrame) -> pd.DataFrame:
    """Deliberately reproduces the bug TIER1_SPEC.md warns about: vx/vy
    get recomputed from the corrupted positions, but speed/direction are
    left untouched at their clean NGS values. Used only in this proof --
    never called from degrade.py's real path."""
    df = df.sort_values(degrade.TRACK_KEY + ["frame_id"]).copy()
    grp = df.groupby(degrade.TRACK_KEY, sort=False)
    key = [df["game_id"], df["play_id"], df["player_id"]]
    vx = (grp["x"].diff() / DT).groupby(key).bfill()
    vy = (grp["y"].diff() / DT).groupby(key).bfill()
    df["vx"] = vx.to_numpy()
    df["vy"] = vy.to_numpy()
    # BUG, on purpose: df["speed"] and df["direction"] are not touched.
    return df


def check_assertion_fires():
    print("=" * 70)
    print("Check 1: does assert_speed_diverged actually fire?")
    print("=" * 70)

    inp, out = load_raw()
    inp, out = normalise(inp, out)
    inp = add_kinematics(inp)

    sigma = 1.0
    rng = np.random.default_rng(DEGRADE_BASE_SEED)
    original_speed = inp["speed"].copy()
    noisy = degrade.add_position_noise(inp, sigma, rng)

    print(f"\n1a. Broken path (speed/direction NOT overwritten), sigma={sigma}:")
    broken = _broken_derive_missing_overwrite(noisy)
    mean_diff_broken = (broken["speed"] - original_speed).abs().mean()
    print(f"    mean |derived_speed - ngs_speed| = {mean_diff_broken:.4f} yd/s "
          f"(should be ~0 -- speed was never touched)")
    try:
        degrade.assert_speed_diverged(original_speed, broken["speed"], sigma)
        print("    FAIL: assertion did not fire -- the guard is broken")
        return False
    except AssertionError as e:
        print(f"    OK: AssertionError raised as expected:\n      {e}")

    print(f"\n1b. Correct path (derive_kinematics_from_position), sigma={sigma}:")
    correct = degrade.derive_kinematics_from_position(noisy)
    mean_diff_correct = (correct["speed"] - original_speed).abs().mean()
    print(f"    mean |derived_speed - ngs_speed| = {mean_diff_correct:.4f} yd/s "
          f"(should be large -- differencing amplifies position noise)")
    degrade.assert_speed_diverged(original_speed, correct["speed"], sigma)
    print("    OK: no exception -- the guard is satisfied on the correct path")
    return True


def check_direction_roundtrip():
    print("\n" + "=" * 70)
    print("Check 2: does the atan2 direction convention round-trip?")
    print("=" * 70)

    inp, out = load_raw()
    inp, out = normalise(inp, out)
    inp = add_kinematics(inp)

    original = inp[["game_id", "play_id", "player_id", "frame_id",
                    "speed", "direction"]].rename(
        columns={"speed": "ngs_speed", "direction": "ngs_direction"})

    # sigma=0: no injected noise. This checks the atan2 convention against
    # real data, not noise robustness (that's check 1 / the full sweep).
    derived = degrade.derive_kinematics_from_position(inp.copy())
    merged = derived.merge(
        original, on=["game_id", "play_id", "player_id", "frame_id"])
    merged = merged.sort_values(degrade.TRACK_KEY + ["frame_id"])

    prev_dir = merged.groupby(degrade.TRACK_KEY)["ngs_direction"].shift(1)
    dir_change = (merged["ngs_direction"] - prev_dir + 180) % 360 - 180
    straight = merged[(merged["ngs_speed"] > 3.0) & (dir_change.abs() < 5.0)]

    angle_diff = (straight["direction"] - straight["ngs_direction"]
                 + 180) % 360 - 180
    abs_diff = angle_diff.abs()

    print(f"\n  straight-line frames checked (NGS speed > 3 yd/s, heading "
          f"change < 5 deg/frame): {len(straight):,}")
    print(f"  mean |derived_direction - ngs_direction|: {abs_diff.mean():.2f} deg")
    print(f"  median: {abs_diff.median():.2f} deg")
    print(f"  95th percentile: {abs_diff.quantile(0.95):.2f} deg")
    print(f"  (a ~90 degree offset would mean atan2(vx, vy) vs "
          f"atan2(vy, vx) is backwards)")

    ok = abs_diff.median() < 10.0
    print(f"  {'OK' if ok else 'FAIL'}: median discrepancy is "
          f"{'small' if ok else 'large'}")
    return ok


def check_mini_sweep():
    print("\n" + "=" * 70)
    print("Check 3: end-to-end mini sweep (MAX_PLAYS=50, sigma=1.0, "
          "both velocity sources)")
    print("=" * 70 + "\n")

    inp, out = load_raw()
    inp, out = normalise(inp, out)
    inp = add_kinematics(inp)
    sigma_a = frozen_sigma_a(inp, out)

    tidy, signed_error = run_sweep([1.0], [0], ["derived", "ngs"], sigma_a,
                                   inp, out, max_plays=50)
    print(f"\n  {len(tidy):,} summary rows produced, "
         f"{len(signed_error):,} signed-error rows produced")
    print(tidy.groupby(["velocity_source", "predictor"])["mde"].mean()
         .sort_values().to_string())
    return True


if __name__ == "__main__":
    ok1 = check_assertion_fires()
    ok2 = check_direction_roundtrip()
    ok3 = check_mini_sweep()
    print("\n" + "=" * 70)
    if ok1 and ok2 and ok3:
        print("ALL SMOKE TESTS PASSED")
    else:
        print("SOME SMOKE TESTS FAILED -- do not run the full sweep")
        raise SystemExit(1)
