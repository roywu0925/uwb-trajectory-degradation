"""The headline figure: K5 vs. all three physics baselines, test split,
1.9s error against sigma, with the crossing marked.

Reads outputs/tier1_test_degrade_summary.csv (committed, small -- the
full test-split degradation sweep across all predictors). The crossing
sigma and its bootstrap CI are NOT recomputed here: they come from the
play-level bootstrap in run_tier1_test_degrade.py (point estimate, 1,000
resamples, seed 0), which needs the >2 MB play-level file this project
doesn't commit (see .gitignore). This is the project's reported crossing
estimate -- see README.md's "Crossing point" line and its Methodology
note; the values are hardcoded below exactly as reported there.

y-axis is clipped to (0, 13): constant_velocity reaches 68.8 yd by
sigma=2.0 and would compress the K5/tbl_kalman_speed crossing -- the
whole point of this figure -- into a sliver at the bottom. The clipped
line is annotated where it exits rather than silently cut off.

    python plot_headline_crossing.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import OUTPUT_DIR

CROSSING_SIGMA = 0.934          # play-level bootstrap point estimate, see README
CROSSING_CI = (0.876, 0.994)    # play-level bootstrap, 1000 resamples, seed 0
YLIM = (0, 13)
ANNOTATION_TOP = 8              # dashed line / CI band scoped to this y, not full axis

# display label -> (code identifier, color, marker, linestyle, linewidth)
# code identifiers are listed in the README caption, not in the legend.
LINES = [
    ("K5 (learned)",                        "K5",                       "#0072B2", "o", "-",  2.5),
    ("Constant velocity",                   "constant_velocity",        "#E69F00", "s", "--", 1.5),
    ("Toward ball-landing point (Kalman speed)", "tbl_kalman_speed",    "#009E73", "^", "-.", 1.5),
    ("Velocity + turn-toward-ball",         "velocity_turning_to_ball", "#D55E00", "D", ":",  1.5),
]
CV_DISPLAY_NAME = LINES[1][0]   # "Constant velocity" -- reused by the
                                # off-scale annotation so it matches the
                                # legend instead of the code identifier


def main():
    df = pd.read_csv(OUTPUT_DIR / "tier1_test_degrade_summary.csv")
    df = df[df["sigma"] <= 2.0].sort_values("sigma").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    frac_top = ANNOTATION_TOP / YLIM[1]
    ax.axvspan(*CROSSING_CI, ymin=0, ymax=frac_top, color="0.3", alpha=0.12, zorder=0)
    ax.axvline(CROSSING_SIGMA, ymin=0, ymax=frac_top, color="0.3", linestyle="--",
              linewidth=1.2)
    ax.text(CROSSING_SIGMA - 0.035, ANNOTATION_TOP * 0.97,
           f"crossing @ σ={CROSSING_SIGMA}\n95% CI [{CROSSING_CI[0]}, {CROSSING_CI[1]}]",
           rotation=90, va="top", ha="right", fontsize=8, color="0.3")

    for label, col, color, marker, ls, lw in LINES:
        ax.plot(df["sigma"], df[col], marker=marker, linestyle=ls, linewidth=lw,
                markersize=5.5, color=color, label=label, clip_on=True)

    # constant_velocity exit point: linear-interpolate sigma where it
    # crosses YLIM[1], annotate instead of leaving it silently clipped
    cv = df["constant_velocity"].to_numpy()
    sigma = df["sigma"].to_numpy()
    exit_i = np.argmax(cv > YLIM[1])
    s0, s1 = sigma[exit_i - 1], sigma[exit_i]
    y0, y1 = cv[exit_i - 1], cv[exit_i]
    exit_sigma = s0 + (YLIM[1] - y0) * (s1 - s0) / (y1 - y0)
    ax.annotate(
        f"{CV_DISPLAY_NAME} → {cv[-1]:.1f} yd at σ={sigma[-1]:.1f} (off scale)",
        xy=(exit_sigma, YLIM[1]), xytext=(1.05, YLIM[1] - 0.6),
        fontsize=8, color="#E69F00", ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color="#E69F00", lw=0.8))

    ax.set_xlim(0, 2.0)
    ax.set_ylim(*YLIM)
    ax.set_xlabel("position noise sigma (yd)")
    ax.set_ylabel("1.9 s mean displacement error (yd)")
    ax.set_title("K5 vs. physics baselines, test split")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()

    path = OUTPUT_DIR / "tier1_test_headline_crossing.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
