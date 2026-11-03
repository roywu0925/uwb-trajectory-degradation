"""Scoring, and the two tables worth looking at.

Horizons vary by play, so a single aggregate number is doubly misleading
here: it hides where physics breaks down, and it mixes short flights with
long ones. Everything below is reported per horizon step, with the number
of plays still contributing at that step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DT, MIN_COVERAGE


def displacement_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Euclidean distance per player per step -> (n_players, horizon)."""
    return np.linalg.norm(pred - truth, axis=-1)


def evaluate(samples: list, predictors: dict) -> pd.DataFrame:
    """Run every predictor over every sample.

    Returns one row per (predictor, play, player, step) collapsed to
    (predictor, step) sums -- kept as tidy per-step records so the caller
    can group by role as well.
    """
    records = []
    for sample in samples:
        roles = (sample.state["role"].to_numpy()
                 if "role" in sample.state.columns
                 else np.full(len(sample.state), "unknown"))
        for name, fn in predictors.items():
            err = displacement_error(fn(sample, sample.horizon),
                                     sample.truth)
            for i, role in enumerate(roles):
                for step in range(sample.horizon):
                    records.append((name, role, step + 1, err[i, step]))

    errors = pd.DataFrame.from_records(
        records, columns=["predictor", "role", "step", "error"])
    errors["seconds"] = errors["step"] * DT
    print(f"scored {len(samples):,} plays, "
          f"{errors['error'].size:,} player-step errors")
    return errors


def curve(errors: pd.DataFrame) -> pd.DataFrame:
    """Mean error and coverage at each horizon step, per predictor."""
    grouped = errors.groupby(["predictor", "step", "seconds"])["error"]
    out = grouped.agg(mean_error="mean", n="size").reset_index()
    max_n = out.groupby("step")["n"].max().max()
    out["coverage"] = out["n"] / max_n
    return out


def summarise(errors: pd.DataFrame) -> pd.DataFrame:
    """One row per predictor, error at 1 s / 2 s / 3 s and overall, each
    tagged with its own track coverage.

    TIER1_SPEC_5.md: `MIN_COVERAGE` existed from Tier 0a onward but was
    only ever wired into plot_error_curves -- this table reported a
    6.40 yd headline at @3s for the entire project without anyone
    checking that @3s coverage was 4.6% on the original sample. Every
    `@Ns_cov` column here is that check, made impossible to skip: a
    `@3s` number with `@3s_cov` below `config.MIN_COVERAGE` is exactly
    the failure mode that produced the wrong headline, and this function
    now makes that visible in the same table rather than a separate plot
    nobody consulted before quoting the number.
    """
    total_tracks = int((errors["step"] == 1).sum() / errors["predictor"].nunique())
    rows = []
    for name, grp in errors.groupby("predictor"):
        by_step = grp.groupby("seconds")["error"].agg(mean="mean", n="size")
        row = {"predictor": name, "overall": grp["error"].mean()}
        for sec in (1.0, 2.0, 3.0):
            hit = by_step.index[np.isclose(by_step.index, sec)]
            if len(hit):
                mean_err, n = by_step.loc[hit[0]]
                row[f"@{sec:.0f}s"] = mean_err
                row[f"@{sec:.0f}s_cov"] = round(n / total_tracks, 3)
        rows.append(row)
    return (pd.DataFrame(rows).sort_values("overall")
            .reset_index(drop=True).round(3))


def summarise_at_step(errors: pd.DataFrame, step: int,
                      min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    """Mean error per predictor at one arbitrary horizon step, with
    (n, coverage) attached to every row and a `trustworthy` column
    gating it against `min_coverage` -- the general form of the coverage
    check `summarise()`'s fixed 1/2/3s columns do inline. Use this for
    any horizon that isn't exactly 1, 2 or 3 seconds (e.g. a data-chosen
    primary horizon), so the guard is never optional.
    """
    total_tracks = int((errors["step"] == 1).sum() / errors["predictor"].nunique())
    at_step = errors[errors["step"] == step]
    out = (at_step.groupby("predictor")["error"].agg(mean_error="mean", n="size")
          .reset_index())
    out["coverage"] = out["n"] / total_tracks
    out["trustworthy"] = out["coverage"] >= min_coverage
    return out.sort_values("mean_error").reset_index(drop=True).round(4)


def summarise_by_role(errors: pd.DataFrame) -> pd.DataFrame:
    """Mean error per predictor per player role.

    More interesting than the headline number: a passer standing still and
    a receiver running a post are not the same prediction problem, and the
    spread across roles says where the remaining error actually lives.
    """
    table = (errors.pivot_table(index="predictor", columns="role",
                                values="error", aggfunc="mean")
             .round(3))
    return table.loc[table.mean(axis=1).sort_values().index]


def plot_error_curves(curves: pd.DataFrame, path):
    """The first real figure of the project."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, grp in curves.groupby("predictor"):
        grp = grp[grp["coverage"] >= MIN_COVERAGE].sort_values("seconds")
        ax.plot(grp["seconds"], grp["mean_error"], label=name, linewidth=1.8)

    ax.set_xlabel("time since throw (s)")
    ax.set_ylabel("mean displacement error (yards)")
    ax.set_title("Physics-only baselines while the ball is in the air")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")
