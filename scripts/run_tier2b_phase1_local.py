"""Tier 2B Phase 1: local inference-time floor, CPU, before touching AWS.

Measures pure inference time for K3, K5 (GRU, torch) and tbl_kalman_speed
(hand-written Kalman filter, pure Python/numpy loop over players) at
batch size 1 and batch size 22 (one play's worth of players). This is
the floor every later network/overhead number sits on top of.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from cache_pipeline import build_dataset_cached
from config import OUTPUT_DIR, WEEKS
from kalman import build_Q, calibrate_sigma_a, estimate_sigma_a
from model import GRUResidualModel, build_examples
from run_tier1_taskk import build_tensors
from run_tier1_taskp import build_predictors_quadrature

N_WARMUP = 20
N_TRIALS = 300


def load_final(variant):
    ckpt = torch.load(OUTPUT_DIR / "checkpoints" / f"{variant}_FINAL_fold0.pt", weights_only=False)
    model = GRUResidualModel(hidden=ckpt["config"]["hidden"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def time_calls(fn, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    for _ in range(n_warmup):
        fn()
    times_ms = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms = np.array(times_ms)
    return float(np.percentile(times_ms, 50)), float(np.percentile(times_ms, 99))


def main():
    torch.set_num_threads(1)   # single-threaded, representative of one serving worker

    samples, _ = build_dataset_cached(WEEKS)
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)
    train_val_samples = [s for s in samples if s.game_id in train_val_games]

    examples = build_examples(train_val_samples)
    tensors = build_tensors(examples)
    n = len(examples)

    k3 = load_final("K3")
    k5 = load_final("K5")

    hist_padded = tensors["hist_padded"]
    hist_lengths = tensors["hist_lengths"]
    static_feat = tensors["static_feat"]

    rng = np.random.default_rng(0)

    def make_batch(batch_size):
        idx = rng.choice(n, size=batch_size, replace=False)
        return hist_padded[idx], hist_lengths[idx], static_feat[idx]

    results = {}
    with torch.no_grad():
        for variant, model in (("K3", k3), ("K5", k5)):
            for batch_size in (1, 22):
                hp, hl, sf = make_batch(batch_size)

                def call():
                    model(hp, hl, sf)

                p50, p99 = time_calls(call)
                results[(variant, batch_size)] = (p50, p99)
                print(f"{variant} batch={batch_size:2d}: p50={p50:.3f} ms  p99={p99:.3f} ms")

    # tbl_kalman_speed -- not a torch model; pure Python/numpy loop over
    # players in sample.state. Build a representative sample with 1 and
    # 22 "players" by reusing real pre-throw histories (timing only,
    # values don't need to be a real single play).
    sigma_a_raw = estimate_sigma_a(train_val_samples)
    sigma_a = calibrate_sigma_a(train_val_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    predictors = build_predictors_quadrature(sigma_a)
    tbl_kalman_speed = predictors["tbl_kalman_speed"]

    import pandas as pd
    rng2 = np.random.default_rng(1)
    sample_idx_pool = train_val_samples

    def make_synthetic_sample(n_players):
        # pick n_players real (player, play) histories and stitch them
        # into one synthetic "play" with n_players players -- same
        # computational shape as a real play, values don't matter for timing
        chosen_plays = rng2.choice(len(sample_idx_pool), size=min(n_players, 8), replace=True)
        state_rows = []
        history_rows = []
        for i, play_idx in enumerate(chosen_plays if n_players <= 8 else
                                      np.tile(chosen_plays, n_players // 8 + 1)[:n_players]):
            s = sample_idx_pool[play_idx]
            row = s.state.iloc[0].copy()
            row["player_id"] = i
            state_rows.append(row)
            h = s.history[s.history.player_id == s.state.iloc[0]["player_id"]].copy()
            h["player_id"] = i
            history_rows.append(h)
        state = pd.DataFrame(state_rows).reset_index(drop=True)
        history = pd.concat(history_rows, ignore_index=True)

        class Synthetic:
            pass
        syn = Synthetic()
        syn.state = state
        syn.history = history
        syn.game_id = 0
        syn.play_id = 0
        return syn

    for n_players in (1, 22):
        syn_sample = make_synthetic_sample(n_players)

        def call():
            tbl_kalman_speed(syn_sample, 19)

        p50, p99 = time_calls(call)
        results[("tbl_kalman_speed", n_players)] = (p50, p99)
        print(f"tbl_kalman_speed batch={n_players:2d}: p50={p50:.3f} ms  p99={p99:.3f} ms")

    print("\n" + "=" * 60)
    print("Phase 1 summary (ms)")
    print("=" * 60)
    print(f"{'model':20s} {'batch':>6s} {'p50':>8s} {'p99':>8s}")
    for (variant, bs), (p50, p99) in results.items():
        print(f"{variant:20s} {bs:6d} {p50:8.3f} {p99:8.3f}")


if __name__ == "__main__":
    main()
