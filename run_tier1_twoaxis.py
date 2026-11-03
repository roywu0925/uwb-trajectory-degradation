"""Tier 1 Task G/H/I: the second degradation axis (TIER1_SPEC_3.md).

    python run_tier1_twoaxis.py

Adds landing-point noise (sigma_ball, on ball_land_x/y) alongside the
existing player-position noise (sigma_player), to find out whether
Finding 4's near-immunity to sigma_player survives a landing point that
is only estimated, not exact -- the hybrid predictors take their entire
direction from that point.

Before the full grid this runs two checks and prints the result of each:

  1. Regression: sigma_ball=0 must reproduce every Part 2 number EXACTLY
     (not approximately), since the two noise axes use independent RNG
     streams and add_ball_land_noise no-ops completely at sigma_ball=0.
  2. The ball-invariance assertion, proven both ways: it must fire on a
     deliberately broken corruption path (ball RNG touching player
     positions) and pass on the correct one.

Only then does it run the 6x6x3 grid plus a few extra points to trace
the sigma_ball=3*sigma_player coupling line (TIER1_SPEC_3.md, Task H),
which the cartesian grid alone barely samples.

Writes outputs/tier1_two_axis.csv, outputs/tier1_two_axis.png,
outputs/tier1_two_axis_heatmap.png.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (BALL_NOISE_BASE_SEED, BALL_SIGMAS, DEGRADE_BASE_SEED, DT,
                    GRID_SEEDS, OUTPUT_DIR, PLAYER_SIGMAS)
from data_loader import build_samples
from degrade import add_ball_land_noise, corrupt_input, corrupt_input_two_axis
from evaluate import evaluate
from imm import make_imm
from run_tier1_sweep import (build_predictors, frozen_sigma_a, load_clean,
                             set_filter_R, summarize_run)

STEP_3S = round(3.0 / DT)
PRIMARY_STEP = 19   # 1.9s, run_tier1_taskn.py's coverage-chosen primary horizon
REPORT_STEPS = (PRIMARY_STEP, STEP_3S)

INVARIANT_PREDICTORS = ["constant_velocity", "kalman", "kalman_velocity",
                        "imm", "kalman_ca"]
DESTINATION_PREDICTORS = ["toward_ball_land", "velocity_turning_to_ball",
                          "tbl_kalman_speed", "tbl_kalman_pos",
                          "tbl_kalman_both"]


def _coverage(step: int) -> tuple[int, float]:
    """(n_tracks, coverage_fraction) at a horizon step, from
    run_tier1_taskn.py's coverage audit -- horizon is a ground-truth
    property, untouched by either noise axis, so this coverage is the
    same regardless of sigma_player, sigma_ball, or predictor."""
    cov = pd.read_csv(OUTPUT_DIR / "horizon_coverage.csv")
    row = cov[cov["step"] == step].iloc[0]
    return int(row["tracks"]), float(row["coverage_tracks"])

# Extra (sigma_player, sigma_ball) pairs to trace sigma_ball=3*sigma_player --
# the 6x6 cartesian grid alone only lands on this line at (0,0) and (1,3).
EXTRA_3X_POINTS = [(0.2, 0.6), (0.5, 1.5), (2.0, 6.0)]


# --------------------------------------------------------- check 1: regression

def regression_check_self_consistent(clean_inp: pd.DataFrame) -> bool:
    """sigma_ball=0 must make corrupt_input_two_axis collapse to plain
    corrupt_input -- exactly, not approximately, since the two axes use
    independent RNG streams and add_ball_land_noise no-ops completely at
    sigma_ball=0.

    This checks the DEGRADED INPUT FRAME directly (no evaluation needed,
    so it's fast) rather than comparing against outputs/tier1_degradation.csv
    as Part 2's version did -- that file is week-1-only and config.WEEKS
    now covers all 18 weeks, so it's no longer a valid reference. The
    mechanism this checks (RNG independence, the sigma_ball<=0 no-op) is
    unchanged from Part 2, where it was already verified against real
    evaluation numbers; this is a cheap re-confirmation, not a new claim.
    """
    ok = True
    for sigma_player in (0.0, 0.2, 1.0, 5.0):
        for seed in (0, 1):
            player_rng_a = np.random.default_rng(DEGRADE_BASE_SEED + seed)
            direct = corrupt_input(clean_inp, sigma_player, player_rng_a, "derived")

            player_rng_b = np.random.default_rng(DEGRADE_BASE_SEED + seed)
            ball_rng = np.random.default_rng(BALL_NOISE_BASE_SEED + seed)
            via_two_axis = corrupt_input_two_axis(
                clean_inp, sigma_player, 0.0, player_rng_b, ball_rng, "derived")

            diff = (direct[["x", "y", "vx", "vy", "speed", "direction"]]
                   - via_two_axis[["x", "y", "vx", "vy", "speed", "direction"]]).abs().max().max()
            print(f"  sigma_player={sigma_player} seed={seed}: max |diff| = {diff:.3e}")
            ok = ok and diff < 1e-9
    print("  EXACT MATCH" if ok else "  MISMATCH -- noise streams are coupled, stop here")
    return ok


# ------------------------------------------------- check 2: invariance guard

def assert_ball_invariant(errors_by_ball_sigma: dict, predictor_names,
                          tol: float = 1e-9) -> None:
    """Predictors in `predictor_names` never read ball_land, so their
    per-(predictor, role, step) mean error must be identical across
    every sigma_ball tested here -- otherwise the two noise streams are
    leaking into each other and the whole grid is suspect.
    """
    sigmas = sorted(errors_by_ball_sigma)
    baseline_sigma = sigmas[0]
    baseline = (errors_by_ball_sigma[baseline_sigma]
               [errors_by_ball_sigma[baseline_sigma]["predictor"].isin(predictor_names)]
               .groupby(["predictor", "role", "step"])["error"].mean())
    for sigma_ball in sigmas[1:]:
        cur = (errors_by_ball_sigma[sigma_ball]
              [errors_by_ball_sigma[sigma_ball]["predictor"].isin(predictor_names)]
              .groupby(["predictor", "role", "step"])["error"].mean())
        diff = (cur - baseline).abs().max()
        assert diff < tol, (
            f"non-destination predictor error changed between "
            f"sigma_ball={baseline_sigma} and sigma_ball={sigma_ball} "
            f"(max diff {diff:.4f}) -- the two noise streams are "
            f"leaking into each other")


def _broken_corrupt_input_two_axis(inp, sigma_player, sigma_ball, player_rng,
                                   ball_rng, velocity_source):
    """Deliberately reproduces a leak -- the ball RNG also perturbs
    player x, y -- used only to prove assert_ball_invariant fires.
    Never called from the real pipeline."""
    df = corrupt_input(inp, sigma_player, player_rng, velocity_source)
    if sigma_ball > 0:
        df = df.copy()
        df["x"] = df["x"] + ball_rng.normal(0, sigma_ball, len(df))
        df["y"] = df["y"] + ball_rng.normal(0, sigma_ball, len(df))
    return add_ball_land_noise(df, sigma_ball, ball_rng)


def prove_invariance_guard(sigma_a, clean_inp, clean_out) -> bool:
    """Small 2x2-ish corner, MAX_PLAYS=50: run the guard against a
    deliberately broken corruption path and show it fires, then against
    the correct one and show it passes."""
    sigma_player = 0.2
    seed = 0
    max_plays = 50

    for label, corrupt_fn in (("broken", _broken_corrupt_input_two_axis),
                              ("correct", corrupt_input_two_axis)):
        errors_by_ball = {}
        for sigma_ball in (0.0, 1.0):
            player_rng = np.random.default_rng(DEGRADE_BASE_SEED + seed)
            ball_rng = np.random.default_rng(BALL_NOISE_BASE_SEED + seed)
            degraded = corrupt_fn(clean_inp, sigma_player, sigma_ball,
                                  player_rng, ball_rng, "derived")
            samples = build_samples(degraded, clean_out)
            samples = samples[:max_plays]
            set_filter_R(sigma_player, "derived")
            predictors = build_predictors(sigma_a)
            errors_by_ball[sigma_ball] = evaluate(samples, predictors)

        print(f"\n  [{label} path] sigma_ball 0.0 vs 1.0, "
             f"sigma_player={sigma_player}:")
        try:
            assert_ball_invariant(errors_by_ball, INVARIANT_PREDICTORS)
            print(f"    PASS: invariant predictors unaffected by sigma_ball")
            if label == "broken":
                print("    FAIL: guard should have fired on the broken path "
                     "but didn't")
                return False
        except AssertionError as e:
            print(f"    RAISED: {e}")
            if label == "correct":
                print("    FAIL: guard should not fire on the correct path")
                return False
            print("    OK: guard fired as expected on the broken path")

    return True


# ------------------------------------------------------------------- grid

def run_two_axis_grid(sigma_a, clean_inp, clean_out, player_sigmas, ball_sigmas,
                      seeds, extra_points=(), max_plays=None
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 6x6xseeds cartesian grid, plus any extra (sigma_player,
    sigma_ball) pairs (for the sigma_ball=3*sigma_player line, which the
    cartesian grid samples at only two points). Runs the ball-invariance
    assertion at every (sigma_player, seed) slice across all sigma_ball
    values tested for it, derived mode only.

    Also folds in the IMM mean-P(manoeuvring)-vs-sigma_player curve at
    sigma_ball=0.0 -- IMM is ball-invariant (it's in INVARIANT_PREDICTORS),
    so its mu trace can't depend on sigma_ball, and this reuses the same
    `samples` already built for that grid point rather than a separate
    sweep. Returns (tidy, mu_df).
    """
    rows = []
    mu_rows = []

    def run_one(sigma_player, sigma_ball, seed, capture_mu=False):
        player_rng = np.random.default_rng(DEGRADE_BASE_SEED + seed)
        ball_rng = np.random.default_rng(BALL_NOISE_BASE_SEED + seed)
        degraded = corrupt_input_two_axis(clean_inp, sigma_player, sigma_ball,
                                          player_rng, ball_rng, "derived")
        samples = build_samples(degraded, clean_out)
        if max_plays is not None and len(samples) > max_plays:
            samples = samples[:max_plays]
        set_filter_R(sigma_player, "derived")
        predictors = build_predictors(sigma_a)
        errors = evaluate(samples, predictors)
        summary = summarize_run(errors, sigma_player, seed, "derived")
        summary = summary.rename(columns={"sigma": "sigma_player"})
        summary["sigma_ball"] = sigma_ball

        if capture_mu:
            mu_log: list = []
            imm_mu_fn = make_imm(sigma_a, mu_log=mu_log)
            for sample in samples:
                imm_mu_fn(sample, sample.horizon)
            mean_mu = (pd.DataFrame(mu_log, columns=["game_id", "play_id",
                                                      "player_id", "frame_id",
                                                      "mu_steady", "mu_manoeuvre"])
                      ["mu_manoeuvre"].mean())
            mu_rows.append((sigma_player, seed, mean_mu, len(mu_log)))

        return errors, summary[["sigma_player", "sigma_ball", "seed", "predictor",
                                "role", "step", "seconds", "mde", "n", "coverage"]]

    n_combos = len(player_sigmas) * len(ball_sigmas) * len(seeds)
    i = 0
    for sigma_player in player_sigmas:
        for seed in seeds:
            errors_by_ball = {}
            for sigma_ball in ball_sigmas:
                i += 1
                errors, summary = run_one(sigma_player, sigma_ball, seed,
                                          capture_mu=(sigma_ball == 0.0))
                errors_by_ball[sigma_ball] = errors
                rows.append(summary)
                print(f"  [{i}/{n_combos}] sigma_player={sigma_player} "
                     f"sigma_ball={sigma_ball} seed={seed}")
            assert_ball_invariant(errors_by_ball, INVARIANT_PREDICTORS)

    print(f"  ball-invariance assertion passed at every "
         f"(sigma_player, seed) slice of the grid")

    for sigma_player, sigma_ball in extra_points:
        for seed in seeds:
            _, summary = run_one(sigma_player, sigma_ball, seed)
            rows.append(summary)
            print(f"  [extra] sigma_player={sigma_player} "
                 f"sigma_ball={sigma_ball} seed={seed}")

    tidy = pd.concat(rows, ignore_index=True)
    mu_df = pd.DataFrame(mu_rows, columns=["sigma_player", "seed",
                                           "mean_mu_manoeuvre", "n_frames"])
    return tidy, mu_df


# --------------------------------------------------------------- sensitivity

def local_slope(tidy: pd.DataFrame, predictor: str, player0: float, ball0: float,
                grid_values: list, axis: str, step: int) -> float | None:
    """Central (or one-sided, at an edge) finite-difference slope of MDE
    at `step`, at (player0, ball0), along `axis` ('player' or 'ball')."""
    at_step = tidy[(tidy["step"] == step) & (tidy["predictor"] == predictor)]

    def mde_at(sigma_player, sigma_ball):
        sub = at_step[(np.isclose(at_step["sigma_player"], sigma_player))
                  & (np.isclose(at_step["sigma_ball"], sigma_ball))]
        if sub.empty:
            return None
        per_seed = sub.groupby("seed").apply(
            lambda g: np.average(g["mde"], weights=g["n"]), include_groups=False)
        return per_seed.mean()

    values = sorted(grid_values)
    idx = values.index(player0 if axis == "player" else ball0)
    lo = values[idx - 1] if idx > 0 else None
    hi = values[idx + 1] if idx < len(values) - 1 else None
    if lo is None or hi is None:
        return None

    if axis == "player":
        y_lo, y_hi = mde_at(lo, ball0), mde_at(hi, ball0)
    else:
        y_lo, y_hi = mde_at(player0, lo), mde_at(player0, hi)
    if y_lo is None or y_hi is None:
        return None
    return (y_hi - y_lo) / (hi - lo)


def report_sensitivity(tidy: pd.DataFrame, step: int, label: str) -> None:
    n_tracks, coverage = _coverage(step)
    print(f"\n--- marginal sensitivity, {label} ({n_tracks} tracks, "
         f"{coverage:.1%} coverage) ---")
    operating_points = [(0.5, 1.0), (1.0, 2.0)]
    for player0, ball0 in operating_points:
        print(f"\nat (sigma_player={player0}, sigma_ball={ball0}):")
        print(f"{'predictor':<26}{'d(mde)/d(sigma_player)':>24}"
             f"{'d(mde)/d(sigma_ball)':>24}{'ratio':>10}")
        for name in DESTINATION_PREDICTORS:
            s_player = local_slope(tidy, name, player0, ball0, PLAYER_SIGMAS, "player", step)
            s_ball = local_slope(tidy, name, player0, ball0, BALL_SIGMAS, "ball", step)
            fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"
            ratio = (f"{abs(s_player / s_ball):.1f}x"
                    if s_player is not None and s_ball not in (None, 0) else "n/a")
            print(f"{name:<26}{fmt(s_player):>24}{fmt(s_ball):>24}{ratio:>10}")


def report_coupled_lines(tidy: pd.DataFrame, step: int, label: str) -> None:
    n_tracks, coverage = _coverage(step)
    print(f"\n--- coupled scenarios, {label} ({n_tracks} tracks, "
         f"{coverage:.1%} coverage) ---")
    at_step = tidy[tidy["step"] == step]

    def mde_grid(predictor, player_vals, ball_vals):
        out = []
        for p, b in zip(player_vals, ball_vals):
            sub = at_step[(at_step["predictor"] == predictor)
                         & (np.isclose(at_step["sigma_player"], p))
                         & (np.isclose(at_step["sigma_ball"], b))]
            if sub.empty:
                out.append(None)
                continue
            per_seed = sub.groupby("seed").apply(
                lambda g: np.average(g["mde"], weights=g["n"]), include_groups=False)
            out.append(round(per_seed.mean(), 3))
        return out

    diag = [v for v in PLAYER_SIGMAS if v in BALL_SIGMAS]
    print(f"\ndiagonal sigma_ball = sigma_player, at {diag}:")
    for name in DESTINATION_PREDICTORS:
        print(f"  {name:<26}{mde_grid(name, diag, diag)}")

    three_x_player = [0.0] + [p for p, b in EXTRA_3X_POINTS] + [1.0]
    three_x_ball = [0.0] + [b for p, b in EXTRA_3X_POINTS] + [3.0]
    order = np.argsort(three_x_player)
    three_x_player = list(np.array(three_x_player)[order])
    three_x_ball = list(np.array(three_x_ball)[order])
    print(f"\nsigma_ball = 3 * sigma_player, at sigma_player = {three_x_player}:")
    for name in DESTINATION_PREDICTORS:
        print(f"  {name:<26}{mde_grid(name, three_x_player, three_x_ball)}")


def report_toward_ball_land_vs_hybrid_on_ball_axis(tidy: pd.DataFrame, step: int,
                                                    label: str) -> None:
    n_tracks, coverage = _coverage(step)
    print(f"\n--- toward_ball_land vs tbl_kalman_speed on the ball axis, "
         f"{label} ({n_tracks} tracks, {coverage:.1%} coverage) ---")
    at_step = tidy[(tidy["step"] == step) & (np.isclose(tidy["sigma_player"], 0.0))]

    def mde_by_ball(predictor):
        sub = at_step[at_step["predictor"] == predictor]
        per_run = sub.groupby(["sigma_ball", "seed"]).apply(
            lambda g: np.average(g["mde"], weights=g["n"]), include_groups=False)
        return per_run.groupby("sigma_ball").mean()

    tbl = mde_by_ball("toward_ball_land")
    hyb = mde_by_ball("tbl_kalman_speed")
    print(pd.DataFrame({"toward_ball_land": tbl, "tbl_kalman_speed": hyb}).round(3)
         .to_string())
    tbl_slope = (tbl.iloc[-1] - tbl.iloc[0]) / (tbl.index[-1] - tbl.index[0])
    hyb_slope = (hyb.iloc[-1] - hyb.iloc[0]) / (hyb.index[-1] - hyb.index[0])
    print(f"end-to-end slope over sigma_ball: toward_ball_land={tbl_slope:.3f}, "
         f"tbl_kalman_speed={hyb_slope:.3f}")


# --------------------------------------------------------------------- figures

def plot_two_axis(tidy: pd.DataFrame, path) -> None:
    import matplotlib.pyplot as plt

    at3s = tidy[tidy["step"] == STEP_3S]
    per_run = (at3s.groupby(["sigma_player", "sigma_ball", "seed", "predictor"])
              .apply(lambda g: np.average(g["mde"], weights=g["n"]),
                     include_groups=False)
              .rename("mde").reset_index())
    summary = (per_run.groupby(["sigma_player", "sigma_ball", "predictor"])["mde"]
              .mean().reset_index())

    predictors = ["tbl_kalman_speed", "toward_ball_land"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for row, name in enumerate(predictors):
        sub = summary[summary["predictor"] == name]

        ax = axes[row, 0]
        for sigma_ball, grp in sub.groupby("sigma_ball"):
            grp = grp.sort_values("sigma_player")
            ax.plot(grp["sigma_player"], grp["mde"], marker="o",
                   label=f"sigma_ball={sigma_ball}")
        ax.set_xlabel("sigma_player (yd)")
        ax.set_ylabel(f"{name}\n3s MDE (yd)")
        ax.set_title(f"{name}: error vs sigma_player")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.legend(frameon=False, fontsize=7)

        ax = axes[row, 1]
        for sigma_player, grp in sub.groupby("sigma_player"):
            grp = grp.sort_values("sigma_ball")
            ax.plot(grp["sigma_ball"], grp["mde"], marker="o",
                   label=f"sigma_player={sigma_player}")
        ax.set_xlabel("sigma_ball (yd)")
        ax.set_ylabel("3s MDE (yd)")
        ax.set_title(f"{name}: error vs sigma_ball")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def plot_heatmap(tidy: pd.DataFrame, path) -> None:
    import matplotlib.pyplot as plt

    at3s = tidy[(tidy["step"] == STEP_3S) & (tidy["predictor"] == "tbl_kalman_speed")]
    per_run = (at3s.groupby(["sigma_player", "sigma_ball", "seed"])
              .apply(lambda g: np.average(g["mde"], weights=g["n"]),
                     include_groups=False)
              .rename("mde").reset_index())
    grid = (per_run.groupby(["sigma_ball", "sigma_player"])["mde"].mean()
           .unstack("sigma_player"))
    grid = grid.reindex(index=sorted(grid.index), columns=sorted(grid.columns))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("sigma_player (yd)")
    ax.set_ylabel("sigma_ball (yd)")
    ax.set_title("tbl_kalman_speed: 3s MDE (yd)")
    fig.colorbar(im, ax=ax, label="3s MDE (yd)")

    def to_pixel(sigma_player, sigma_ball):
        xs = list(grid.columns)
        ys = list(grid.index)
        x = np.interp(sigma_player, xs, range(len(xs)))
        y = np.interp(sigma_ball, ys, range(len(ys)))
        return x, y

    diag_p = [v for v in PLAYER_SIGMAS if v in BALL_SIGMAS]
    diag_xy = [to_pixel(p, p) for p in diag_p]
    ax.plot([xy[0] for xy in diag_xy], [xy[1] for xy in diag_xy],
           "w--", linewidth=2, label="sigma_ball = sigma_player")

    three_x_p = sorted([0.0] + [p for p, b in EXTRA_3X_POINTS] + [1.0])
    three_x_xy = [to_pixel(p, 3 * p) for p in three_x_p if 3 * p <= max(BALL_SIGMAS)]
    ax.plot([xy[0] for xy in three_x_xy], [xy[1] for xy in three_x_xy],
           "r--", linewidth=2, label="sigma_ball = 3 x sigma_player")

    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------- main

def main():
    import time
    t_start = time.time()

    print("loading full 18-week dataset...")
    clean_inp, clean_out = load_clean()
    sigma_a = frozen_sigma_a(clean_inp, clean_out)

    print("\n" + "=" * 70)
    print("Check 1: sigma_ball=0 collapses corrupt_input_two_axis to corrupt_input")
    print("=" * 70)
    ok1 = regression_check_self_consistent(clean_inp)
    if not ok1:
        raise SystemExit("regression check failed -- stopping before the grid")

    print("\n" + "=" * 70)
    print("Check 2: ball-invariance guard, proven both ways")
    print("=" * 70)
    ok2 = prove_invariance_guard(sigma_a, clean_inp, clean_out)
    if not ok2:
        raise SystemExit("invariance guard proof failed -- stopping before the grid")

    print("\nboth checks passed -- running the full grid on all 18 weeks\n")
    n_combos = len(PLAYER_SIGMAS) * len(BALL_SIGMAS) * GRID_SEEDS
    print(f"running two-axis grid: {len(PLAYER_SIGMAS)} player sigmas x "
         f"{len(BALL_SIGMAS)} ball sigmas x {GRID_SEEDS} seeds = "
         f"{n_combos} runs, plus {len(EXTRA_3X_POINTS)} extra points "
         f"x {GRID_SEEDS} seeds for the 3x coupling line -- capturing "
         f"steps {REPORT_STEPS} (1.9s primary, 3.0s secondary) and the "
         f"IMM mu trace at sigma_ball=0 in the same pass")
    tidy, mu_df = run_two_axis_grid(sigma_a, clean_inp, clean_out, PLAYER_SIGMAS,
                                    BALL_SIGMAS, range(GRID_SEEDS),
                                    extra_points=EXTRA_3X_POINTS)

    tidy.to_csv(OUTPUT_DIR / "tier1_two_axis.csv", index=False)
    print(f"\nwrote {OUTPUT_DIR / 'tier1_two_axis.csv'} ({len(tidy):,} rows)")
    mu_df.to_csv(OUTPUT_DIR / "tier1_two_axis_imm_mu.csv", index=False)
    print(f"wrote {OUTPUT_DIR / 'tier1_two_axis_imm_mu.csv'} ({len(mu_df):,} rows)")

    print(f"\ntotal grid time: {time.time() - t_start:.0f}s")

    for step, label in ((PRIMARY_STEP, "primary (1.9s)"), (STEP_3S, "secondary (3.0s)")):
        print("\n" + "#" * 70)
        print(f"# {label}")
        print("#" * 70)
        report_sensitivity(tidy, step, label)
        report_coupled_lines(tidy, step, label)
        report_toward_ball_land_vs_hybrid_on_ball_axis(tidy, step, label)

    print("\n" + "=" * 70)
    print("IMM mean P(manoeuvring) vs sigma_player (sigma_ball=0, ball-invariant)")
    print("=" * 70)
    print(mu_df.groupby("sigma_player")["mean_mu_manoeuvre"].agg(["mean", "std"])
         .round(4).to_string())

    plot_two_axis(tidy, OUTPUT_DIR / "tier1_two_axis.png")
    plot_heatmap(tidy, OUTPUT_DIR / "tier1_two_axis_heatmap.png")


if __name__ == "__main__":
    main()
