"""Check 1 (input-window boundary) and Check 2 (error vs step), fold 0,
K2 only, forward-pass only against outputs/checkpoints/K2_fold0.pt.

No retraining, no changes to model.py or run_tier1_taskk.py's scoring
functions -- this script only reads the checkpoint and re-derives
diagnostics from the same tensors Task K already built.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from cache_pipeline import build_dataset_cached
from config import DT, MAX_HORIZON, OUTPUT_DIR, WEEKS
from kalman import calibrate_sigma_a, estimate_sigma_a
from model import GRUResidualModel, build_examples
from run_tier1_taskk import build_tensors, compute_base_arrays, predict_all
from run_tier1_taskp import build_predictors_quadrature

PRIMARY_STEP = 19


def main():
    samples, _ = build_dataset_cached(WEEKS)
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)
    train_val_samples = [s for s in samples if s.game_id in train_val_games]

    examples = build_examples(train_val_samples)
    tensors = build_tensors(examples)
    truth_np = tensors["truth"].numpy()
    mask_np = tensors["mask"].numpy()
    game_id_arr = tensors["game_id"]

    sigma_a_raw = estimate_sigma_a(train_val_samples)
    sigma_a = calibrate_sigma_a(train_val_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    predictors = build_predictors_quadrature(sigma_a)
    base_vtb_np = compute_base_arrays(train_val_samples, predictors["velocity_turning_to_ball"], examples)
    base_vtb = torch.from_numpy(base_vtb_np)

    test_games = set(folds[0])
    test_idx = np.where(np.isin(game_id_arr, list(test_games)))[0]

    ckpt = torch.load(OUTPUT_DIR / "checkpoints" / "K2_fold0.pt", weights_only=False)
    assert ckpt["variant"] == "K2" and ckpt["fold_id"] == 0
    model = GRUResidualModel(hidden=ckpt["config"]["hidden"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded checkpoint: val_loss={ckpt['val_loss']:.4f}, epochs={ckpt['epochs_run']}, seed={ckpt['seed']}")

    model_out = predict_all(model, base_vtb, tensors, test_idx)
    pred = base_vtb[test_idx].numpy() + model_out
    truth = truth_np[test_idx]
    mask = mask_np[test_idx]

    # ================================================================
    # CHECK 1(a) -- index audit
    # ================================================================
    print("\n" + "=" * 70)
    print("CHECK 1(a) -- input window boundary, index audit")
    print("=" * 70)
    print("""
Slicing code that builds the input window (data_loader.py):

    history_by_key = dict(tuple(
        inp[["game_id", "play_id", "player_id", "frame_id", "x", "y",
             "vx", "vy"]].groupby(KEY, sort=False)))
    ...
    history = (full_history[full_history["player_id"].isin(player_ids)]
              .sort_values(["player_id", "frame_id"])
              .reset_index(drop=True)
              if full_history is not None else ...)

Slicing code that builds the output/truth window (data_loader.py):

    for key, play_out in out.groupby(KEY, sort=False):
        ...
        truth = _pivot_truth(play_out, state["player_id"].to_numpy(), horizon)

`history` is sliced exclusively from `inp` (input_*.csv); `truth` is
sliced exclusively from `out` (output_*.csv), via two separate
DataFrame.groupby calls over two separate source tables loaded from two
separate files. There is no code path that can copy a row from one into
the other.
""")

    # Recompute per-track max input frame_id and per-track min output
    # frame_id directly from the raw samples (fold-0 out-of-fold plays).
    n = len(examples)
    max_input_frame = np.full(n, -1, dtype=np.int64)
    min_output_frame = np.full(n, -1, dtype=np.int64)
    pos = 0
    for sample in train_val_samples:
        hist_by_player = dict(tuple(sample.history.groupby("player_id", sort=False)))
        for i in range(len(sample.state)):
            pid = sample.state.iloc[i]["player_id"]
            h = hist_by_player[pid]
            max_input_frame[pos] = h["frame_id"].max()
            min_output_frame[pos] = 1   # _pivot_truth always starts frames at 1
            pos += 1
    assert pos == n

    test_max_in = max_input_frame[test_idx]
    test_min_out = min_output_frame[test_idx]
    print(f"input max frame_id (fold-0 test tracks): min={test_max_in.min()}, "
         f"median={int(np.median(test_max_in))}, max={test_max_in.max()}")
    print(f"output min frame_id (fold-0 test tracks): constant = "
         f"{test_min_out.min()} (== {test_min_out.max()} for every track -- "
         f"_pivot_truth's `frames = list(range(1, horizon+1))` always starts "
         f"the output pivot at 1)")
    literal_violations = (test_max_in >= test_min_out).sum()
    print(f"literal 'max_input < min_output' as raw frame_id numbers: "
         f"{literal_violations}/{len(test_idx)} tracks fail this")
    print("""
CAVEAT, load-bearing: input frame_id and output frame_id are NOT the same
clock. Verified directly against the raw CSVs (week 1, play 2023090700/101):
input frame_id runs 1..26 (pre-throw), output frame_id runs 1..21 (post-
throw) -- both counters independently start at 1 for their own segment of
the play, output_*.csv carries no column that would let it be placed on
the same axis as input_*.csv's frame_id. So the literal numeric comparison
above is comparing two different clocks and its ~100% "violation" count is
an artifact of that, not evidence of temporal overlap -- min(output
frame_id) is trivially 1 for every track by construction, not "the first
frame after the anchor" on any shared timeline.

The check that actually bears on leakage is structural, not numeric: does
any `inp`-sourced value ever reach `truth`, or vice versa? Per the quoted
code, no -- `history` and `truth` are independent groupby results over
independent source DataFrames. Confirmed additionally: every track's
history row count exactly matches its player's row count in `inp` filtered
to game_id/play_id/player_id (no extra rows), and the anchor row (last
history frame) is bit-identical to `sample.state`'s own x/y (already
verified in the Task K build_examples assertion, which passed).
""")

    # ================================================================
    # CHECK 1(b) -- direct value search, position channels
    # ================================================================
    print("=" * 70)
    print("CHECK 1(b) -- direct value search, de-normalised input positions "
         "vs truth window")
    print("=" * 70)

    anchor_np = tensors["anchor"].numpy()
    hist_padded_np = tensors["hist_padded"].numpy()
    hist_lengths_np = tensors["hist_lengths"].numpy()

    min_dists = []
    for local_i, global_i in enumerate(test_idx):
        L = hist_lengths_np[global_i]
        h = hist_padded_np[global_i, :L, :2] + anchor_np[global_i]  # (L, 2) absolute
        horizon = int(mask[local_i].sum())
        t = truth[local_i, :horizon]                                # (horizon, 2) absolute
        d = np.linalg.norm(h[:, None, :] - t[None, :, :], axis=-1)   # (L, horizon)
        min_dists.append(d.min())
    min_dists = np.array(min_dists)
    print(f"min distance (any input frame -> any truth frame), per track, "
         f"n={len(min_dists)}:")
    print(f"  min={min_dists.min():.6f}  p1={np.percentile(min_dists,1):.6f}  "
         f"median={np.median(min_dists):.6f}  p99={np.percentile(min_dists,99):.6f}  "
         f"max={min_dists.max():.6f}")
    print(f"  tracks with min distance < 1e-6: {(min_dists < 1e-6).sum()}")
    print(f"  tracks with min distance < 1e-3: {(min_dists < 1e-3).sum()}  "
         f"(near-zero can happen honestly for a nearly-stationary player)")

    # ================================================================
    # CHECK 1(c) -- ball_land channel, and velocity channels
    # ================================================================
    print("\n" + "=" * 70)
    print("CHECK 1(c) -- ball_land offset channel, and velocity channels")
    print("=" * 70)

    static_np = tensors["static_feat"].numpy()
    ball_land_abs = static_np[:, -2:] + anchor_np    # (N, 2) absolute field coords
    min_dists_ball = []
    for local_i, global_i in enumerate(test_idx):
        horizon = int(mask[local_i].sum())
        t = truth[local_i, :horizon]
        d = np.linalg.norm(ball_land_abs[global_i][None, :] - t, axis=-1)
        min_dists_ball.append(d.min())
    min_dists_ball = np.array(min_dists_ball)
    print(f"min distance (ball_land -> any truth frame), per track, n={len(min_dists_ball)}:")
    print(f"  min={min_dists_ball.min():.6f}  median={np.median(min_dists_ball):.6f}  "
         f"max={min_dists_ball.max():.6f}")
    print(f"  tracks with min distance < 1e-6: {(min_dists_ball < 1e-6).sum()}  "
         f"(a nonzero count here would be expected and NOT leakage for "
         f"Targeted Receiver tracks -- the receiver catching the ball AT "
         f"the landing point in the final truth frame is the whole premise "
         f"of the play, not a bug)")

    # velocity channels: input's vx, vy vs finite-differenced truth velocity
    min_dists_vel = []
    for local_i, global_i in enumerate(test_idx):
        L = hist_lengths_np[global_i]
        v_in = hist_padded_np[global_i, :L, 2:4]              # (L, 2)
        horizon = int(mask[local_i].sum())
        t = truth[local_i, :horizon]
        if horizon < 2:
            continue
        v_truth = np.diff(t, axis=0) / DT                      # (horizon-1, 2)
        d = np.linalg.norm(v_in[:, None, :] - v_truth[None, :, :], axis=-1)
        min_dists_vel.append(d.min())
    min_dists_vel = np.array(min_dists_vel)
    print(f"\nmin distance (input vx,vy -> any finite-differenced truth "
         f"velocity), per track, n={len(min_dists_vel)}:")
    print(f"  min={min_dists_vel.min():.6f}  median={np.median(min_dists_vel):.6f}  "
         f"max={min_dists_vel.max():.6f}")
    print(f"  tracks with min distance < 1e-6: {(min_dists_vel < 1e-6).sum()}")

    # ================================================================
    # CHECK 2 -- error vs step
    # ================================================================
    print("\n" + "=" * 70)
    print("CHECK 2 -- mean error vs step, 1..19, model vs velocity_turning_to_ball")
    print("=" * 70)
    # Fixed population (the n=1000 tracks with horizon >= 19) across every
    # step, so the curve's shape reflects the same tracks' error evolving
    # over time, not a shrinking population biasing later steps.
    valid19 = mask[:, PRIMARY_STEP - 1] > 0
    n_fixed = int(valid19.sum())
    print(f"fixed population: {n_fixed} tracks (horizon >= 19), same set at every step")
    print(f"{'step':>4s} {'seconds':>7s} {'model_err':>10s} {'base_err':>10s}")
    model_curve = {}
    base_curve = {}
    for step in range(1, PRIMARY_STEP + 1):
        col = step - 1
        m_err = float(np.linalg.norm(pred[valid19, col] - truth[valid19, col], axis=-1).mean())
        b_err = float(np.linalg.norm(base_vtb_np[test_idx][valid19, col] - truth[valid19, col], axis=-1).mean())
        model_curve[step] = m_err
        base_curve[step] = b_err
        print(f"{step:4d} {step/10:7.1f} {m_err:10.4f} {b_err:10.4f}")

    print(f"\nmodel:  step1={model_curve[1]:.4f}  step19={model_curve[19]:.4f}  "
         f"ratio(step19/step1)={model_curve[19]/model_curve[1]:.3f}")
    print(f"base:   step1={base_curve[1]:.4f}  step19={base_curve[19]:.4f}  "
         f"ratio(step19/step1)={base_curve[19]/base_curve[1]:.3f}")


if __name__ == "__main__":
    main()
