"""Draw one play: where the tracked players actually went, and where a
baseline thought they would go.

This is the README cover image, and the fastest way to catch a broken
coordinate transform -- if the mirroring is wrong you see it at a glance,
and no metric would have told you.
"""

from __future__ import annotations

import numpy as np

from config import DT, FIELD_WIDTH

TRUTH_C = "#1d3f6e"
PRED_C = "#d85a30"
CONTEXT_C = "#9a9a93"
BALL_C = "#1d9e75"


def draw_field(ax, x_min: float, x_max: float):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, FIELD_WIDTH)
    ax.set_aspect("equal")
    ax.set_facecolor("#f3f6f1")
    start = int(x_min) - int(x_min) % 5
    for yard in range(start, int(x_max) + 1, 5):
        ax.axvline(yard, color="white", zorder=0,
                   linewidth=1.4 if yard % 10 == 0 else 0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def animate_play(sample, pred: np.ndarray, path, label: str = "prediction",
                 title: str = ""):
    """sample: PlaySample. pred: (n_players, horizon, 2)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    truth = sample.truth
    horizon = sample.horizon
    ball = sample.state[["ball_land_x", "ball_land_y"]].to_numpy()[0]

    xs = np.concatenate([truth[..., 0].ravel(), pred[..., 0].ravel(),
                         sample.context["x"].to_numpy(), [ball[0]]])
    pad = 5.0
    x_min, x_max = xs.min() - pad, xs.max() + pad

    fig, ax = plt.subplots(figsize=(9, 9 * FIELD_WIDTH / (x_max - x_min)))
    draw_field(ax, x_min, x_max)
    if title:
        ax.set_title(title, fontsize=11, pad=8)

    others = sample.context[~sample.context["player_to_predict"]]
    ax.scatter(others["x"], others["y"], s=42, c=CONTEXT_C, alpha=0.45,
               zorder=2, label="other players (at throw)")
    ax.scatter(*ball, marker="X", s=150, c=BALL_C, zorder=5,
               label="ball landing spot")

    n = len(sample.state)
    truth_dots = ax.scatter(truth[:, 0, 0], truth[:, 0, 1], s=80, c=TRUTH_C,
                            zorder=4, label="actual")
    pred_dots = ax.scatter(pred[:, 0, 0], pred[:, 0, 1], s=80,
                           facecolors="none", edgecolors=PRED_C,
                           linewidths=1.8, zorder=4, label=label)
    links = [ax.plot([], [], color=PRED_C, linewidth=1.0, alpha=0.6,
                     zorder=3)[0] for _ in range(n)]
    trails = [ax.plot([], [], color=TRUTH_C, linewidth=1.2, alpha=0.35,
                      zorder=2)[0] for _ in range(n)]
    clock = ax.text(0.015, 0.96, "", transform=ax.transAxes, fontsize=10,
                    va="top", color="#333")
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    def update(step):
        truth_dots.set_offsets(truth[:, step, :])
        pred_dots.set_offsets(pred[:, step, :])
        for i in range(n):
            links[i].set_data([truth[i, step, 0], pred[i, step, 0]],
                              [truth[i, step, 1], pred[i, step, 1]])
            trails[i].set_data(truth[i, :step + 1, 0], truth[i, :step + 1, 1])
        err = np.linalg.norm(truth[:, step] - pred[:, step], axis=-1).mean()
        clock.set_text(f"t = +{(step + 1) * DT:.1f} s     "
                       f"mean error = {err:.2f} yd")
        return [truth_dots, pred_dots, clock, *links, *trails]

    anim = FuncAnimation(fig, update, frames=horizon, interval=100, blit=False)
    anim.save(path, writer=PillowWriter(fps=10), dpi=110)
    plt.close(fig)
    print(f"wrote {path}")
