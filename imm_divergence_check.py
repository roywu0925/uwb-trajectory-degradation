"""IMM divergence check: does the manoeuvring model collapse under noise?

    python imm_divergence_check.py

After the Task C R fix, kalman_ca got *worse* at sigma=5 (54.34 -> 59.57)
while imm improved sharply (138.66 -> 39.37). Both share the same 6-dim
constant-acceleration state and, since Task C, the same R -- the only
structural difference is that imm can shift weight away from the
higher-process-noise "manoeuvring" model onto the "steady" one, and
kalman_ca cannot (it always runs at Q_steady).

Hypothesis: at high sensor noise the double-differenced acceleration
state becomes useless, and IMM's model probabilities detect that and
collapse onto "steady" -- which, with its much smaller Q, behaves far
more conservatively (trusts its own smooth prediction over a wildly
noisy per-step update) than "manoeuvring" does. kalman_ca has no such
escape and is stuck running as if every frame might be a real manoeuvre.

If true, this reframes Tier 0b's IMM result rather than contradicting
it: the switching mechanism isn't for detecting real motion regimes (it
won 0.0% of frames on clean data, Tier 0b/CONTEXT.md) -- it's
robustness against a state that becomes unusable under sensor
degradation. Test: plot mean P(manoeuvring) against sigma, derived mode,
full 0-5 range. Collapse toward 0 as sigma grows supports the
hypothesis; a flat line does not, and the kalman_ca/imm divergence needs
another explanation.

Writes outputs/imm_mu_vs_sigma.csv and outputs/imm_model_prob_vs_sigma.png.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DEGRADE_BASE_SEED, N_SEEDS, NOISE_SIGMAS, OUTPUT_DIR
from data_loader import build_samples
from degrade import corrupt_input
from imm import make_imm
from run_tier1_sweep import frozen_sigma_a, load_clean, set_filter_R


def collect_mu_vs_sigma(sigma_a: float, clean_inp: pd.DataFrame,
                        clean_out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sigma in NOISE_SIGMAS:
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(DEGRADE_BASE_SEED + seed)
            degraded_inp = corrupt_input(clean_inp, sigma, rng, "derived")
            samples = build_samples(degraded_inp, clean_out)

            set_filter_R(sigma, "derived")
            mu_log: list = []
            imm_fn = make_imm(sigma_a, mu_log=mu_log)
            for sample in samples:
                imm_fn(sample, sample.horizon)

            mu_df = pd.DataFrame(mu_log, columns=[
                "game_id", "play_id", "player_id", "frame_id",
                "mu_steady", "mu_manoeuvre"])
            mean_mu = mu_df["mu_manoeuvre"].mean()
            rows.append((sigma, seed, mean_mu, len(mu_df)))
            print(f"  sigma={sigma} seed={seed}: mean P(manoeuvring) = "
                 f"{mean_mu:.4f} over {len(mu_df):,} frames")

    return pd.DataFrame(rows, columns=["sigma", "seed", "mean_mu_manoeuvre",
                                       "n_frames"])


def plot_mu_vs_sigma(df: pd.DataFrame, path) -> None:
    import matplotlib.pyplot as plt

    summary = (df.groupby("sigma")["mean_mu_manoeuvre"]
              .agg(mean="mean", std="std").reset_index())
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.errorbar(summary["sigma"], summary["mean"], yerr=summary["std"].fillna(0),
               marker="o", linewidth=1.8, capsize=3, color="tab:purple")
    ax.set_xlabel("position noise sigma (yd)")
    ax.set_ylabel("mean P(manoeuvring)")
    ax.set_title("IMM manoeuvring-model probability vs. sensor noise "
                 "(derived mode)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    clean_inp, clean_out = load_clean()
    sigma_a = frozen_sigma_a(clean_inp, clean_out)

    df = collect_mu_vs_sigma(sigma_a, clean_inp, clean_out)
    df.to_csv(OUTPUT_DIR / "imm_mu_vs_sigma.csv", index=False)
    plot_mu_vs_sigma(df, OUTPUT_DIR / "imm_model_prob_vs_sigma.png")

    print("\nmean P(manoeuvring) by sigma, averaged over seeds:")
    print(df.groupby("sigma")["mean_mu_manoeuvre"].mean().round(4).to_string())
