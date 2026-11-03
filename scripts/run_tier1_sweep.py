"""Tier 1 Task B: the sensor degradation sweep (TIER1_SPEC.md).

    PYTHONPATH=. python scripts/run_tier1_sweep.py  # run from the repo root

Corrupts input_*.csv x, y with controlled Gaussian noise, re-derives
downstream kinematics from the corrupted positions (or doesn't -- see
config.VELOCITY_SOURCES), and re-runs baselines.py's six physics
predictors plus the Tier 0b filters against the SAME, uncorrupted
output_*.csv ground truth. No new predictors, no training, no changes to
baselines.py or kalman.py.

R tracks the sensor, block diagonal -- position and velocity noise set
independently, not scaled together as a single isotropic sigma (see
set_filter_R; this was fixed per TIER1_SPEC_2.md Task C after Task B's
original R = max(sigma,0.05)**2 * I scaled velocity with position sigma
in both modes, which is wrong in both). Q does not track the sensor at
all -- it is estimated once on the clean data, exactly as Tier 0b did,
and held fixed at every noise level, because Q describes how
unpredictably players move (a property of the players) and R describes
measurement noise (a property of the sensor). kalman.py and imm.py keep R
as a module-level constant referenced inside their predict/update loops
rather than a per-call parameter, so varying it here means reassigning
that module attribute from outside -- not editing either file, but worth
being explicit about, since it looks unusual on a first read.

Writes outputs/tier1_degradation.csv and outputs/tier1_robustness.png.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import imm as imm_module
import kalman as kalman_module
from baselines import BASELINES as PHYSICS_BASELINES
from config import (DEGRADE_BASE_SEED, DEGRADE_SIGMA_VEL_NGS, DT, N_SEEDS,
                    NOISE_SIGMAS, OUTPUT_DIR, VELOCITY_SOURCES)
from data_loader import add_kinematics, build_samples, load_raw, normalise
from degrade import corrupt_input
from evaluate import evaluate
from hybrid import make_hybrid_family
from imm import make_imm, make_kalman_ca
from kalman import build_Q, calibrate_sigma_a, estimate_sigma_a, make_kalman


def load_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same load -> normalise -> add_kinematics steps data_loader.
    build_dataset uses, stopped one step early so degrade.corrupt_input
    can be inserted before build_samples."""
    inp, out = load_raw()
    inp, out = normalise(inp, out)
    inp = add_kinematics(inp)
    return inp, out


def frozen_sigma_a(clean_inp: pd.DataFrame, clean_out: pd.DataFrame) -> float:
    """Q's scale, estimated once on the clean dataset exactly as Tier 0b
    did (raw differencing estimate -> NIS calibration), then held fixed
    across the whole sweep. Not re-estimated per noise level -- see the
    module docstring."""
    clean_samples = build_samples(clean_inp, clean_out)
    sigma_a_raw = estimate_sigma_a(clean_samples)
    sigma_a = calibrate_sigma_a(clean_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    print(f"frozen Tier 0b sigma_a = {sigma_a:.3f} yd/s^2 (from clean "
          f"data, raw estimate was {sigma_a_raw:.3f}; reused unchanged "
          f"at every noise level)")
    return sigma_a


def set_filter_R(sigma: float, velocity_source: str) -> None:
    """Block-diagonal R: position and velocity noise set independently
    (TIER1_SPEC_2.md, Task C).

    Task B's R = max(sigma, 0.05)**2 * I scaled every observed dimension
    with position sigma, in both modes -- wrong both ways: in "derived"
    mode it understated the true finite-differencing noise by ~14x,
    letting the filter over-trust an almost-pure-noise measurement; in
    "ngs" mode it inflated R on a velocity channel that was never
    actually degraded.

    sigma_pos: max(sigma, 0.05), unchanged from Task B.

    sigma_vel:
      "derived" -- sqrt(2) * sigma_pos / dt. The velocity "measurement"
        here is (x[t] - x[t-1]) / dt, a difference of two noisy position
        samples each with variance sigma_pos**2, so its variance is
        2 * sigma_pos**2 / dt**2.
      "ngs" -- config.DEGRADE_SIGMA_VEL_NGS, constant and independent of
        sigma, since NGS's s/dir columns are never corrupted in this
        mode.

    Known approximation, accepted rather than fixed: finite-differenced
    velocities at consecutive frames share a position sample (v[t] and
    v[t+1] both use x[t]), so they are correlated -- violating the
    white-measurement-noise assumption the Kalman update makes. Noted
    here and in the write-up, not corrected.
    """
    sigma_pos = max(sigma, 0.05)
    if velocity_source == "derived":
        sigma_vel = np.sqrt(2) * sigma_pos / DT
    elif velocity_source == "ngs":
        sigma_vel = DEGRADE_SIGMA_VEL_NGS
    else:
        raise ValueError(f"unknown velocity_source {velocity_source!r}")

    r_pos_vel = np.diag([sigma_pos**2, sigma_pos**2, sigma_vel**2, sigma_vel**2])
    kalman_module.R_POS = (sigma_pos ** 2) * np.eye(2)
    kalman_module.R_POS_VEL = r_pos_vel
    imm_module.R = r_pos_vel


def build_predictors(sigma_a: float) -> dict:
    Q = build_Q(sigma_a)
    predictors = dict(PHYSICS_BASELINES)
    predictors["kalman"] = make_kalman(Q, use_velocity=False)
    predictors["kalman_velocity"] = make_kalman(Q, use_velocity=True)
    predictors["kalman_ca"] = make_kalman_ca(sigma_a)
    predictors["imm"] = make_imm(sigma_a)
    predictors.update(make_hybrid_family(Q))
    return predictors


SIGNED_ERROR_PREDICTORS = ["toward_ball_land", "kalman", "kalman_velocity",
                          "tbl_kalman_speed", "tbl_kalman_pos", "tbl_kalman_both"]
STEP_3S = round(3.0 / DT)


def compute_signed_error(samples, predictors: dict, names, step: int) -> list:
    """Mean signed along-track error at a fixed step, for a subset of
    predictors (hybrid.py's module docstring has the motivation: MDE
    can't see a systematic undershoot, only magnitude).

    predicted_direction = normalize(pred - raw anchor position): the
    direction from the known anchor to that predictor's own predicted
    point at this step, defined identically for every predictor so the
    metric is comparable across the whole family, including ones (like
    kalman) that have no destination vector of their own.

    signed_error = dot(pred - true, predicted_direction): negative means
    the prediction falls short of where the player actually ended up
    (undershoot), positive means it overshoots, near zero means no
    systematic bias along that predictor's own claimed direction.
    """
    rows = []
    for sample in samples:
        if sample.horizon < step:
            continue
        anchor = sample.state[["x", "y"]].to_numpy()
        true_pos = sample.truth[:, step - 1, :]
        for name in names:
            pred = predictors[name](sample, sample.horizon)[:, step - 1, :]
            direction = pred - anchor
            dist = np.linalg.norm(direction, axis=-1, keepdims=True)
            unit = np.divide(direction, dist, out=np.zeros_like(direction),
                            where=dist > 1e-6)
            signed = np.sum((pred - true_pos) * unit, axis=-1)
            for value in signed:
                rows.append((name, float(value)))
    return rows


def summarize_run(errors: pd.DataFrame, sigma: float, seed: int,
                  velocity_source: str) -> pd.DataFrame:
    """One row per (predictor, role, step) -> mde, n, coverage. Mirrors
    evaluate.curve()'s coverage convention (n relative to the largest n
    seen at any step in this run) but keeps role as its own dimension,
    per TIER1_SPEC.md's deliverable shape."""
    grouped = errors.groupby(["predictor", "role", "step"])["error"]
    out = grouped.agg(mde="mean", n="size").reset_index()
    max_n = out.groupby("step")["n"].max().max()
    out["coverage"] = out["n"] / max_n
    out["seconds"] = out["step"] * DT
    out["sigma"] = sigma
    out["seed"] = seed
    out["velocity_source"] = velocity_source
    return out[["sigma", "seed", "velocity_source", "predictor", "role",
               "step", "seconds", "mde", "n", "coverage"]]


def run_sweep(sigmas, seeds, velocity_sources, sigma_a: float,
             clean_inp: pd.DataFrame, clean_out: pd.DataFrame,
             max_plays: int | None = None
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The full (sigma, seed, velocity_source) loop. `max_plays`
    subsamples plays for a fast smoke test; leave None for the real
    sweep, which uses every play config.py's MIN_HORIZON/MAX_HORIZON/
    WEEKS already select (unchanged from Tier 0/0b).

    Returns (tidy, signed_error): `tidy` is the aggregated MDE table;
    `signed_error` is one row per (sigma, seed, velocity_source,
    predictor, signed_error) for SIGNED_ERROR_PREDICTORS at the 3s
    horizon -- see compute_signed_error.
    """
    rows = []
    signed_rows = []
    combos = list(itertools.product(sigmas, seeds, velocity_sources))
    for i, (sigma, seed, velocity_source) in enumerate(combos, 1):
        rng = np.random.default_rng(DEGRADE_BASE_SEED + seed)
        degraded_inp = corrupt_input(clean_inp, sigma, rng, velocity_source)
        samples = build_samples(degraded_inp, clean_out)
        if max_plays is not None and len(samples) > max_plays:
            samples = samples[:max_plays]

        set_filter_R(sigma, velocity_source)
        predictors = build_predictors(sigma_a)

        errors = evaluate(samples, predictors)
        rows.append(summarize_run(errors, sigma, seed, velocity_source))

        for name, value in compute_signed_error(
                samples, predictors, SIGNED_ERROR_PREDICTORS, STEP_3S):
            signed_rows.append((sigma, seed, velocity_source, name, value))

        print(f"  [{i}/{len(combos)}] sigma={sigma} seed={seed} "
              f"velocity_source={velocity_source}: "
              f"{len(samples)} plays scored")

    tidy = pd.concat(rows, ignore_index=True)
    signed_df = pd.DataFrame(signed_rows, columns=[
        "sigma", "seed", "velocity_source", "predictor", "signed_error"])
    return tidy, signed_df


def plot_robustness(tidy: pd.DataFrame, path, target_seconds: float = 3.0,
                    caption: str | None = None) -> None:
    """3 s MDE against sigma, one line per predictor, error bars across
    seeds, one panel per velocity_source."""
    import matplotlib.pyplot as plt

    step = round(target_seconds / DT)
    at_horizon = tidy[tidy["step"] == step]

    # role-weighted overall MDE per (sigma, seed, velocity_source, predictor)
    per_run = (at_horizon.groupby(["sigma", "seed", "velocity_source", "predictor"])
              .apply(lambda g: np.average(g["mde"], weights=g["n"]),
                     include_groups=False)
              .rename("mde").reset_index())
    across_seeds = (per_run.groupby(["sigma", "velocity_source", "predictor"])["mde"]
                    .agg(mean="mean", std="std").reset_index())

    sources = sorted(across_seeds["velocity_source"].unique())
    fig, axes = plt.subplots(1, len(sources), figsize=(7 * len(sources), 5),
                             sharey=True)
    if len(sources) == 1:
        axes = [axes]

    for ax, source in zip(axes, sources):
        panel = across_seeds[across_seeds["velocity_source"] == source]
        for name, grp in panel.groupby("predictor"):
            grp = grp.sort_values("sigma")
            ax.errorbar(grp["sigma"], grp["mean"], yerr=grp["std"].fillna(0),
                       label=name, marker="o", linewidth=1.6, capsize=3)
        ax.set_xlabel("position noise sigma (yd)")
        ax.set_title(f"velocity_source = {source}")
        ax.set_yscale("log")
        ax.grid(alpha=0.25, linewidth=0.5, which="both")
        ax.spines[["top", "right"]].set_visible(False)

    # log scale because constant_acceleration's double-differencing
    # blow-up under velocity_source=derived spans four orders of
    # magnitude (~6 to ~7500 yd) -- a linear axis makes every other
    # predictor, and the entire "ngs" panel, read as a flat line at zero.
    axes[0].set_ylabel(f"{target_seconds:.0f} s mean displacement error "
                       f"(yd, log scale)")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    if caption:
        fig.subplots_adjust(bottom=0.24)
        fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=8,
                 wrap=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading clean dataset...")
    clean_inp, clean_out = load_clean()
    sigma_a = frozen_sigma_a(clean_inp, clean_out)

    print(f"\nrunning sweep: {len(NOISE_SIGMAS)} sigmas x {N_SEEDS} seeds x "
          f"{len(VELOCITY_SOURCES)} velocity sources = "
          f"{len(NOISE_SIGMAS) * N_SEEDS * len(VELOCITY_SOURCES)} runs")
    tidy, signed_error = run_sweep(NOISE_SIGMAS, range(N_SEEDS), VELOCITY_SOURCES,
                                  sigma_a, clean_inp, clean_out)

    tidy.to_csv(OUTPUT_DIR / "tier1_degradation.csv", index=False)
    print(f"\nwrote {OUTPUT_DIR / 'tier1_degradation.csv'} ({len(tidy):,} rows)")

    signed_error.to_csv(OUTPUT_DIR / "tier1_signed_error_full.csv", index=False)
    print(f"wrote {OUTPUT_DIR / 'tier1_signed_error_full.csv'} "
          f"({len(signed_error):,} rows)")

    plot_robustness(tidy, OUTPUT_DIR / "tier1_robustness.png")

    # sigma=0.0 "derived" is the honest left endpoint of the new curve --
    # print it explicitly next to what Tier 0 reported, since they will
    # not match (TIER1_SPEC.md).
    print("\nsigma=0.0, velocity_source=derived, 3s MDE (compare against "
          "Tier 0's numbers in README.md -- expected to differ):")
    zero = tidy[(tidy["sigma"] == 0.0) & (tidy["velocity_source"] == "derived")
               & (tidy["step"] == round(3.0 / DT))]
    per_pred = (zero.groupby("predictor")
               .apply(lambda g: np.average(g["mde"], weights=g["n"]),
                      include_groups=False)
               .sort_values())
    print(per_pred.to_string())

    print("\nmean signed along-track error at 3s (negative = undershoot):")
    print(signed_error.groupby(["velocity_source", "sigma", "predictor"])
         ["signed_error"].mean().unstack("predictor").round(3).to_string())


if __name__ == "__main__":
    main()
