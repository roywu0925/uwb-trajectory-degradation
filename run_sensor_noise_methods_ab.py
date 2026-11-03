"""Sensor noise measurement, methods A (difference-order ladder) and B
(power spectrum), dropping the stationarity requirement entirely. Every
tracked frame of every play, no role filter, no window selection.
"""

from __future__ import annotations

import json
from math import comb

import numpy as np
import pandas as pd

from config import DT, OUTPUT_DIR, WEEKS
from data_loader import TRAIN_DIR

ORDERS = [1, 2, 3, 4]


def load_raw_tracks(games: set) -> dict:
    """(game_id, play_id, nfl_id) -> sorted (x, y) arrays, every frame."""
    cols = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y"]
    parts = []
    for week in WEEKS:
        for path in sorted(TRAIN_DIR.glob(f"input_*_{week}.csv")):
            df = pd.read_csv(path, usecols=cols)
            df = df[df["game_id"].isin(games)]
            if not df.empty:
                parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    tracks = {}
    for key, grp in full.groupby(["game_id", "play_id", "nfl_id"], sort=False):
        grp = grp.sort_values("frame_id")
        tracks[key] = (grp["x"].to_numpy(), grp["y"].to_numpy())
    return tracks


# ================================================================ Method A
def nth_diff(v: np.ndarray, n: int) -> np.ndarray:
    out = v
    for _ in range(n):
        out = np.diff(out)
    return out


def method_a_sigma(tracks: dict, axis_idx: int, order: int) -> tuple[float, np.ndarray]:
    diffs = []
    for (x, y) in tracks.values():
        v = x if axis_idx == 0 else y
        if len(v) > order:
            diffs.append(nth_diff(v, order))
    diffs = np.concatenate(diffs) if diffs else np.array([])
    divisor = np.sqrt(comb(2 * order, order))
    return float(np.std(diffs) / divisor), diffs


def method_a_bootstrap(tracks_by_play: dict, order: int, axis_idx: int,
                       n_resamples: int = 1000, seed: int = 0) -> np.ndarray:
    play_keys = list(tracks_by_play.keys())
    n_plays = len(play_keys)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n_plays, size=n_plays)
        diffs = []
        for i in idx:
            for (x, y) in tracks_by_play[play_keys[i]].values():
                v = x if axis_idx == 0 else y
                if len(v) > order:
                    diffs.append(nth_diff(v, order))
        diffs = np.concatenate(diffs)
        divisor = np.sqrt(comb(2 * order, order))
        out.append(float(np.std(diffs) / divisor))
    return np.array(out)


def group_by_play(tracks: dict) -> dict:
    by_play = {}
    for (gid, pid, nid), v in tracks.items():
        by_play.setdefault((gid, pid), {})[(gid, pid, nid)] = v
    return by_play


# ================================================================ Method B
def welch_like_psd(tracks: dict, axis_idx: int, n_freq_bins: int = 50, fmax: float = 5.0):
    """Per-track: linear detrend, Hann window, periodogram. Interpolate
    each onto a common frequency grid (track lengths vary), then average
    across all tracks -- the Welch-average step, done across independent
    plays instead of within one long signal."""
    freq_grid = np.linspace(0.05, fmax, n_freq_bins)
    interpolated = []
    for (x, y) in tracks.values():
        v = x if axis_idx == 0 else y
        n = len(v)
        if n < 6:
            continue
        t = np.arange(n)
        coeffs = np.polyfit(t, v, 1)
        detrended = v - np.polyval(coeffs, t)
        window = np.hanning(n)
        windowed = detrended * window
        # normalise so the periodogram integrates to the windowed signal's
        # power (standard Welch/periodogram scaling)
        win_power = np.sum(window ** 2)
        fft_vals = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(n, d=DT)
        psd = (np.abs(fft_vals) ** 2) / (win_power * (1.0 / DT))
        # one-sided: double all but DC/Nyquist
        psd[1:-1] *= 2
        valid = freqs <= fmax
        if freqs[valid].max() < fmax * 0.5:
            continue   # too short to see the band of interest at all
        interp_vals = np.interp(freq_grid, freqs[valid], psd[valid])
        interpolated.append(interp_vals)
    if not interpolated:
        return freq_grid, None
    interpolated = np.array(interpolated)
    return freq_grid, interpolated.mean(axis=0)


def main():
    with open(OUTPUT_DIR / "final_val_games.json") as f:
        fv = json.load(f)
    train_games = set(int(g) for g in fv["final_train_games"]) | set(int(g) for g in fv["final_val_games"])
    with open(OUTPUT_DIR / "splits.json") as f:
        splits = json.load(f)
    test_games = set(splits["test"])

    print("loading raw tracks, training games, every frame, every player...")
    train_tracks = load_raw_tracks(train_games)
    print(f"{len(train_tracks):,} tracks")
    train_by_play = group_by_play(train_tracks)
    print(f"{len(train_by_play):,} plays")

    # ---------------------------------------------------------- Method A
    print("\n" + "=" * 70)
    print("METHOD A -- difference-order ladder, TRAINING, every frame/player")
    print("=" * 70)
    rows = []
    for order in ORDERS:
        for axis_idx, axis_name in enumerate(("x", "y")):
            sigma_hat, diffs = method_a_sigma(train_tracks, axis_idx, order)
            boot = method_a_bootstrap(train_by_play, order, axis_idx, n_resamples=500)
            ci = np.percentile(boot, [2.5, 97.5])
            rows.append({"order": order, "axis": axis_name, "sigma_hat": sigma_hat,
                        "ci_lo": ci[0], "ci_hi": ci[1], "n": len(diffs)})
            print(f"  order={order} axis={axis_name}: sigma_hat={sigma_hat:.4f} "
                 f"95% CI=({ci[0]:.4f}, {ci[1]:.4f})  n={len(diffs):,}")
        # pooled across axes
        diffs_x = method_a_sigma(train_tracks, 0, order)[1]
        diffs_y = method_a_sigma(train_tracks, 1, order)[1]
        pooled = np.concatenate([diffs_x, diffs_y])
        divisor = np.sqrt(comb(2 * order, order))
        pooled_sigma = float(np.std(pooled) / divisor)
        print(f"  order={order} pooled (x+y): sigma_hat={pooled_sigma:.4f}")
        rows.append({"order": order, "axis": "pooled", "sigma_hat": pooled_sigma,
                    "ci_lo": np.nan, "ci_hi": np.nan, "n": len(pooled)})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "sensor_noise_method_a.csv", index=False)

    print("\nconvergence check (pooled, orders 2,3,4 vs each other):")
    pooled_vals = df[(df["axis"] == "pooled")].set_index("order")["sigma_hat"]
    for o in [2, 3, 4]:
        if o in pooled_vals.index and (o - 1) in pooled_vals.index:
            delta_pct = (pooled_vals[o] - pooled_vals[o - 1]) / pooled_vals[o - 1] * 100
            print(f"  order {o-1}->{o}: {pooled_vals[o-1]:.4f} -> {pooled_vals[o]:.4f} ({delta_pct:+.1f}%)")

    print("\nTEST GAMES -- consistency check only")
    test_tracks = load_raw_tracks(test_games)
    test_rows = []
    for order in ORDERS:
        diffs_x = method_a_sigma(test_tracks, 0, order)[1]
        diffs_y = method_a_sigma(test_tracks, 1, order)[1]
        pooled = np.concatenate([diffs_x, diffs_y])
        divisor = np.sqrt(comb(2 * order, order))
        pooled_sigma = float(np.std(pooled) / divisor)
        print(f"  order={order} pooled: sigma_hat={pooled_sigma:.4f}")
        test_rows.append({"order": order, "sigma_hat": pooled_sigma})
    pd.DataFrame(test_rows).to_csv(OUTPUT_DIR / "sensor_noise_method_a_test.csv", index=False)

    # ---------------------------------------------------------- Method B
    print("\n" + "=" * 70)
    print("METHOD B -- power spectrum, TRAINING")
    print("=" * 70)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    psd_summary = {}
    for axis_idx, axis_name in enumerate(("x", "y")):
        freq_grid, psd_mean = welch_like_psd(train_tracks, axis_idx)
        if psd_mean is None:
            print(f"  {axis_name}: not enough long tracks for a PSD estimate")
            continue
        high_band = (freq_grid >= 3.0) & (freq_grid <= 5.0)
        floor = float(np.median(psd_mean[high_band]))
        sigma_from_psd = float(np.sqrt(floor * (1.0 / DT) / 2.0))
        print(f"  {axis_name}: high-band (3-5Hz) median PSD floor = {floor:.6f} yd^2/Hz, "
             f"implied sigma = {sigma_from_psd:.4f} yd")
        psd_summary[axis_name] = sigma_from_psd

        ax = axes[axis_idx]
        ax.loglog(freq_grid, psd_mean)
        ax.axhline(floor, color="red", linestyle="--", label=f"3-5Hz floor")
        ax.axvspan(3.0, 5.0, alpha=0.15, color="red")
        ax.set_xlabel("Hz")
        ax.set_ylabel("PSD (yd^2/Hz)")
        ax.set_title(f"{axis_name}: averaged periodogram, {len(train_tracks):,} tracks")
        ax.legend()

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "sensor_noise_psd.png"
    plt.savefig(fig_path, dpi=120)
    print(f"\nsaved {fig_path}")

    print(f"\nMethod A (order-4, pooled) vs Method B:")
    a4 = pooled_vals.get(4, np.nan)
    print(f"  A: {a4:.4f}  B: x={psd_summary.get('x', float('nan')):.4f} "
         f"y={psd_summary.get('y', float('nan')):.4f}")


if __name__ == "__main__":
    main()
