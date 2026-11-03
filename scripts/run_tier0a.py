"""Tier 0a + 0b end to end.

    PYTHONPATH=. python scripts/run_tier0a.py  # run from the repo root

Writes to outputs/:
    baseline_errors.csv     per-predictor, per-step, per-role errors
    error_curves.png        error against time since the throw
    error_by_role.csv       which roles each baseline handles badly
    play_animation.gif      one play, actual against a baseline

The Kalman baselines (Tier 0b) are registered here rather than statically
in baselines.py because their process-noise Q is estimated from the loaded
dataset -- see kalman.estimate_sigma_a. It runs once per dataset, not once
per play, and is never fit against output_*.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from animate import animate_play
from baselines import BASELINES, constant_velocity
from config import DT, IMM_STEADY_SIGMA_A_EXPERIMENT, OUTPUT_DIR
from data_loader import build_dataset
from evaluate import (curve, evaluate, plot_error_curves, summarise,
                      summarise_by_role)
from imm import make_imm, make_kalman_ca
from kalman import (build_Q, calibrate_sigma_a, compute_mean_nis,
                    estimate_sigma_a, make_kalman)

ANIMATE_BASELINE = "constant_velocity"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = build_dataset()

    sigma_a_raw = estimate_sigma_a(samples)
    print(f"kalman: sigma_a from differencing NGS velocity = "
          f"{sigma_a_raw:.3f} yd/s^2 (diagnostic only -- known biased "
          f"high, differencing a noisy signal amplifies its noise)")
    nis_raw, _ = compute_mean_nis(samples, build_Q(sigma_a_raw), use_velocity=False)
    print(f"  NIS at that sigma_a = {nis_raw:.2f} (expected ~= 2) -- "
          f"confirms the bias: Q is too large, the filter is more "
          f"uncertain than it should be")

    sigma_a = calibrate_sigma_a(samples, sigma_a_hi=sigma_a_raw, target=2.0)
    Q = build_Q(sigma_a)
    print(f"  NIS-calibrated sigma_a = {sigma_a:.3f} yd/s^2 -- this is "
          f"what the filter actually uses (answer-blind: calibrated "
          f"against the filter's own innovation consistency, never "
          f"output_*.csv)")
    for name, use_velocity, dim_z in (("kalman", False, 2),
                                      ("kalman_velocity", True, 4)):
        nis, n = compute_mean_nis(samples, Q, use_velocity)
        print(f"  {name}: mean NIS = {nis:.2f} over {n:,} update steps "
              f"(expected ~= {dim_z} if Q, R are well specified)")

    kalman_fn = make_kalman(Q, use_velocity=False)
    kalman_vel_fn = make_kalman(Q, use_velocity=True)
    BASELINES["kalman"] = kalman_fn
    BASELINES["kalman_velocity"] = kalman_vel_fn

    # IMM: two 6-dim constant-acceleration models sharing the same state,
    # differing only in process noise -- "steady" uses the same
    # NIS-calibrated sigma_a as kalman/kalman_velocity above, "manoeuvring"
    # is IMM_MANOEUVRE_RATIO times larger (config.py, a fixed design
    # choice, not fit to this dataset).
    mu_log: list = []
    imm_fn = make_imm(sigma_a, mu_log=mu_log)
    BASELINES["imm"] = imm_fn

    # Experiment A: does IMM's gain over kalman_velocity come from regime
    # switching, or just from the richer 6-dim (explicit-acceleration)
    # state? A standalone run of the steady model alone, no mixing.
    kalman_ca_fn = make_kalman_ca(sigma_a)
    BASELINES["kalman_ca"] = kalman_ca_fn

    # Experiment B: re-specify the steady model's sigma_a (see the
    # comment on IMM_STEADY_SIGMA_A_EXPERIMENT in config.py for why
    # 1.791 was the wrong value for a "non-manoeuvring" model). One
    # shot, reported alongside `imm`, not replacing it.
    mu_log_b: list = []
    imm_resigma_fn = make_imm(IMM_STEADY_SIGMA_A_EXPERIMENT, mu_log=mu_log_b)
    BASELINES["imm_resigma"] = imm_resigma_fn

    print("evaluating baselines...")
    errors = evaluate(samples, BASELINES)

    for name, fn in (("kalman", kalman_fn), ("kalman_velocity", kalman_vel_fn),
                     ("imm", imm_fn), ("kalman_ca", kalman_ca_fn),
                     ("imm_resigma", imm_resigma_fn)):
        n = fn.short_history_count()
        if n:
            print(f"  {name}: {n} player-track(s) had fewer than "
                  f"KALMAN_MIN_HISTORY frames, fell back to constant "
                  f"velocity from the anchor state")

    for label, log, path in (
        ("imm", mu_log, "imm_model_probabilities.csv"),
        ("imm_resigma", mu_log_b, "imm_model_probabilities_resigma.csv"),
    ):
        mu_df = pd.DataFrame(log, columns=["game_id", "play_id", "player_id",
                                           "frame_id", "mu_steady",
                                           "mu_manoeuvre"])
        mu_df.to_csv(OUTPUT_DIR / path, index=False)
        frac_manoeuvre = (mu_df["mu_manoeuvre"] > 0.5).mean()
        print(f"{label}: {frac_manoeuvre:.1%} of scored pre-throw frames "
              f"favour the manoeuvring model (wrote outputs/{path})")
        if frac_manoeuvre < 0.02 or frac_manoeuvre > 0.98:
            print(f"  -- that's within 2% of the boundary, which most "
                  f"likely means the two models aren't meaningfully "
                  f"discriminating, not that motion is genuinely that "
                  f"one-sided; see CONTEXT.md")

    errors.to_csv(OUTPUT_DIR / "baseline_errors.csv", index=False)

    print("\nmean displacement error, yards (lower is better):")
    print(summarise(errors).to_string(index=False))

    by_role = summarise_by_role(errors)
    by_role.to_csv(OUTPUT_DIR / "error_by_role.csv")
    print("\nmean error by player role:")
    print(by_role.to_string())
    print()

    plot_error_curves(curve(errors), OUTPUT_DIR / "error_curves.png")

    sample = max(samples, key=lambda s: s.horizon)
    pred = BASELINES[ANIMATE_BASELINE](sample, sample.horizon)
    err = np.linalg.norm(sample.truth - pred, axis=-1).mean()
    animate_play(
        sample, pred, OUTPUT_DIR / "play_animation.gif",
        label=ANIMATE_BASELINE.replace("_", " "),
        title=f"game {sample.game_id}, play {sample.play_id} -- "
              f"{sample.horizon * DT:.1f} s of ball flight, "
              f"mean error {err:.2f} yd")


if __name__ == "__main__":
    main()
