"""Closing items 1 and 3, plus the per-role baseline table, all forward-pass
or pure-baseline computation against outputs/checkpoints/K2_fold0.pt.
No retraining here (that's the separate permutation-control script).
"""

from __future__ import annotations

import json

import numpy as np
import torch

from cache_pipeline import build_dataset_cached
from config import DT, OUTPUT_DIR, WEEKS
from kalman import calibrate_sigma_a, estimate_sigma_a
from model import GRUResidualModel, build_examples
from run_tier1_taskk import build_tensors, compute_base_arrays, masked_error_at_step, predict_all
from run_tier1_taskp import build_predictors_quadrature

PRIMARY_STEP = 19
PRIMARY_COL = PRIMARY_STEP - 1


def main():
    samples, _ = build_dataset_cached(WEEKS)
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)
    train_val_samples = [s for s in samples if s.game_id in train_val_games]

    examples = build_examples(train_val_samples)
    roles = np.array([e.role for e in examples])
    tensors = build_tensors(examples)
    truth_np = tensors["truth"].numpy()
    mask_np = tensors["mask"].numpy()
    game_id_arr = tensors["game_id"]
    anchor_np = tensors["anchor"].numpy()
    hist_padded_np = tensors["hist_padded"].numpy()
    hist_lengths_np = tensors["hist_lengths"].numpy()

    sigma_a_raw = estimate_sigma_a(train_val_samples)
    sigma_a = calibrate_sigma_a(train_val_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    predictors = build_predictors_quadrature(sigma_a)

    needed = ["velocity_turning_to_ball", "constant_velocity", "kalman",
             "tbl_kalman_speed", "toward_ball_land"]
    base_arrays = {name: compute_base_arrays(train_val_samples, predictors[name], examples)
                  for name in needed}

    test_games = set(folds[0])
    val_games = set(folds[1])
    train_games = train_val_games - test_games - val_games
    test_idx = np.where(np.isin(game_id_arr, list(test_games)))[0]
    train_idx = np.where(np.isin(game_id_arr, list(train_games)))[0]

    ckpt = torch.load(OUTPUT_DIR / "checkpoints" / "K2_fold0.pt", weights_only=False)
    model = GRUResidualModel(hidden=ckpt["config"]["hidden"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    base_vtb_t = torch.from_numpy(base_arrays["velocity_turning_to_ball"])

    # ================================================================
    # CLOSING ITEM 1 -- the 3 velocity near-matches, full precision
    # ================================================================
    print("=" * 70)
    print("CLOSING ITEM 1 -- the 3 velocity near-matches, full detail")
    print("=" * 70)

    # rebuild the (track, input_frame, truth_frame) triples exactly as
    # in check 1(c), but keep the argmin location this time
    n = len(examples)
    example_meta = []   # (game_id, play_id, player_id) per flattened index
    sample_by_key = {}
    for sample in train_val_samples:
        sample_by_key[(sample.game_id, sample.play_id)] = sample
        for i in range(len(sample.state)):
            row = sample.state.iloc[i]
            example_meta.append((sample.game_id, sample.play_id, row["player_id"]))
    assert len(example_meta) == n

    found = 0
    for global_i in test_idx:
        L = hist_lengths_np[global_i]
        v_in = hist_padded_np[global_i, :L, 2:4]
        horizon = int(mask_np[global_i].sum())
        t = truth_np[global_i, :horizon]
        if horizon < 2:
            continue
        v_truth = np.diff(t, axis=0) / DT
        d = np.linalg.norm(v_in[:, None, :] - v_truth[None, :, :], axis=-1)
        min_d = d.min()
        if min_d < 1e-6:
            found += 1
            ii, jj = np.unravel_index(np.argmin(d), d.shape)
            gid, pid, plid = example_meta[global_i]
            sample = sample_by_key[(gid, pid)]
            hist_sorted = (sample.history[sample.history.player_id == plid]
                          .sort_values("frame_id"))
            input_frame_id = int(hist_sorted.iloc[ii]["frame_id"])
            print(f"\ntrack {found}: game={gid} play={pid} player={plid}")
            print(f"  input frame_id (pre-throw) = {input_frame_id} "
                 f"(row {ii} of {len(hist_sorted)} history rows), "
                 f"vs truth-derived velocity between output frames {jj+1} and {jj+2} "
                 f"(post-throw, {(jj+1)*DT:.1f}s-{(jj+2)*DT:.1f}s)")
            print(f"  input vx, vy (full precision): {v_in[ii,0]!r}, {v_in[ii,1]!r}")
            print(f"  truth-derived velocity (full precision): {v_truth[jj,0]!r}, {v_truth[jj,1]!r}")
            s_implied = float(np.hypot(*v_in[ii]))
            dir_implied = float(np.degrees(np.arctan2(v_in[ii,0], v_in[ii,1])) % 360)
            s_truth = float(np.hypot(*v_truth[jj]))
            dir_truth = float(np.degrees(np.arctan2(v_truth[jj,0], v_truth[jj,1])) % 360)
            print(f"  implied (s, dir) from input vx,vy:   s={s_implied!r}, dir={dir_implied!r}")
            print(f"  implied (s, dir) from truth velocity: s={s_truth!r}, dir={dir_truth!r}")
            print(f"  |v_in - v_truth| = {min_d!r}")
    print(f"\ntotal found: {found}")

    # ================================================================
    # CLOSING ITEM 3 -- train-fold error alongside out-of-fold, K2 fold0
    # ================================================================
    print("\n" + "=" * 70)
    print("CLOSING ITEM 3 -- train-fold vs out-of-fold error, K2 fold0")
    print("=" * 70)
    print(f"model parameter count: {model.count_parameters():,} "
         f"(not ~100k -- correcting the estimate)")
    print(f"train examples this fold: {len(train_idx):,}")

    model_out_test = predict_all(model, base_vtb_t, tensors, test_idx)
    pred_test = base_vtb_t[test_idx].numpy() + model_out_test
    err_test, n_test = masked_error_at_step(pred_test, truth_np[test_idx], mask_np[test_idx], PRIMARY_STEP)

    model_out_train = predict_all(model, base_vtb_t, tensors, train_idx)
    pred_train = base_vtb_t[train_idx].numpy() + model_out_train
    err_train, n_train = masked_error_at_step(pred_train, truth_np[train_idx], mask_np[train_idx], PRIMARY_STEP)

    print(f"train-fold error @1.9s:      {err_train:.4f}  (n={n_train})")
    print(f"out-of-fold error @1.9s:     {err_test:.4f}  (n={n_test})")
    print(f"gap (oof - train):           {err_test - err_train:.4f}")

    # ================================================================
    # SUBSTANTIVE -- per-role baseline table, fold 0 out-of-fold, @1.9s
    # ================================================================
    print("\n" + "=" * 70)
    print("Per-role baseline table, fold 0 out-of-fold, @1.9s")
    print("=" * 70)
    valid = mask_np[test_idx, PRIMARY_COL] > 0
    r = roles[test_idx][valid]
    model_err_per_track = np.linalg.norm(
        pred_test[valid, PRIMARY_COL] - truth_np[test_idx][valid, PRIMARY_COL], axis=-1)

    per_role_rows = {}
    for role in ["Defensive Coverage", "Targeted Receiver"]:
        m = r == role
        row = {"role": role, "n": int(m.sum())}
        best_name, best_val_ = None, np.inf
        for name in needed:
            arr = base_arrays[name][test_idx][valid]
            e = np.linalg.norm(arr[m, PRIMARY_COL] - truth_np[test_idx][valid][m, PRIMARY_COL], axis=-1).mean()
            row[name] = e
            if e < best_val_:
                best_val_, best_name = e, name
        row["best_baseline"] = best_name
        row["best_baseline_err"] = best_val_
        row["K2_model"] = model_err_per_track[m].mean()
        per_role_rows[role] = row

    header = ["role", "n"] + needed + ["best_baseline", "best_baseline_err", "K2_model"]
    print(" | ".join(header))
    for role, row in per_role_rows.items():
        vals = [f"{row[h]:.4f}" if isinstance(row.get(h), float) else str(row.get(h)) for h in header]
        print(" | ".join(vals))

    # overall (both roles combined) -- reproduces the two headline numbers
    print("\n--- overall (both roles combined, fold 0 out-of-fold) ---")
    vtb_overall = base_arrays["velocity_turning_to_ball"][test_idx][valid]
    vtb_err_overall = np.linalg.norm(
        vtb_overall[:, PRIMARY_COL] - truth_np[test_idx][valid, PRIMARY_COL], axis=-1).mean()
    model_err_overall = model_err_per_track.mean()

    n_dc, n_tr = per_role_rows["Defensive Coverage"]["n"], per_role_rows["Targeted Receiver"]["n"]
    best_overall_weighted = (per_role_rows["Defensive Coverage"]["best_baseline_err"] * n_dc +
                             per_role_rows["Targeted Receiver"]["best_baseline_err"] * n_tr) / (n_dc + n_tr)

    print(f"model vs velocity_turning_to_ball:                {model_err_overall:.4f} vs {vtb_err_overall:.4f}  "
         f"(margin {100*(1 - model_err_overall/vtb_err_overall):.1f}%)")
    print(f"model vs BEST-PER-ROLE baseline (role-weighted):  {model_err_overall:.4f} vs {best_overall_weighted:.4f}  "
         f"(margin {100*(1 - model_err_overall/best_overall_weighted):.1f}%)")
    print(f"  DC best baseline: {per_role_rows['Defensive Coverage']['best_baseline']} "
         f"({per_role_rows['Defensive Coverage']['best_baseline_err']:.4f})")
    print(f"  TR best baseline: {per_role_rows['Targeted Receiver']['best_baseline']} "
         f"({per_role_rows['Targeted Receiver']['best_baseline_err']:.4f})")


if __name__ == "__main__":
    main()
