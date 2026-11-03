"""Check B -- label-permutation negative control. One training run.

Shuffle (truth, mask, base) across fold 0's train+val tracks only, so
every track's input FEATURES stay its own but its TRAINING TARGET
becomes some other, unrelated track's target. Train K2 on that exactly
as normal, then score it out-of-fold on the REAL, unpermuted test set.

If the reported 1.82 yd @1.9s result depends on genuine feature->target
structure, a model trained on permuted targets should be unable to beat
even a naive baseline on the real test set (best it can do is learn the
population's rough average displacement). If it can still somehow score
well, that points at a leak somewhere in the harness rather than at
genuine learning.

Checkpointed separately from the real run, as
outputs/checkpoints/K2_PERMUTED_CONTROL_fold0.pt -- does not touch or
overwrite outputs/checkpoints/K2_fold0.pt.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from cache_pipeline import build_dataset_cached
from config import OUTPUT_DIR, SPLIT_SEED, WEEKS
from kalman import calibrate_sigma_a, estimate_sigma_a
from model import build_examples
from run_tier1_taskk import (build_tensors, compute_base_arrays,
                             masked_error_at_step, predict_all, train_one)
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

    test_games = set(folds[0])
    val_games = set(folds[1])
    train_games = train_val_games - test_games - val_games
    test_idx = np.where(np.isin(game_id_arr, list(test_games)))[0]
    val_idx = np.where(np.isin(game_id_arr, list(val_games)))[0]
    train_idx = np.where(np.isin(game_id_arr, list(train_games)))[0]

    seed = SPLIT_SEED  # same seed convention as fold 0's real run
    rng = np.random.default_rng(seed)
    trainval_idx = np.concatenate([train_idx, val_idx])
    shuffled = rng.permutation(trainval_idx)
    n_self_matches = int((shuffled == trainval_idx).sum())
    print(f"permuting {len(trainval_idx):,} train+val tracks' targets "
         f"({n_self_matches} unavoidable self-matches from the random permutation)")

    truth_permuted = truth_np.copy()
    mask_permuted = mask_np.copy()
    base_permuted = base_vtb_np.copy()
    truth_permuted[trainval_idx] = truth_np[shuffled]
    mask_permuted[trainval_idx] = mask_np[shuffled]
    base_permuted[trainval_idx] = base_vtb_np[shuffled]
    # test_idx entries are untouched -- real targets, used only for scoring

    tensors_permuted = dict(tensors)
    tensors_permuted["truth"] = torch.from_numpy(truth_permuted)
    tensors_permuted["mask"] = torch.from_numpy(mask_permuted)
    base_permuted_t = torch.from_numpy(base_permuted)

    model, best_val, wall_clock, epochs_run = train_one(
        "K2_PERMUTED_CONTROL", base_permuted_t, tensors_permuted,
        train_idx, val_idx, seed, fold_id=0, checkpoint=True)
    print(f"trained on permuted targets: val_loss={best_val:.4f} "
         f"epochs={epochs_run} wall_clock={wall_clock:.1f}s")

    # score on the REAL, unpermuted test set -- base and truth both
    # come from the original (non-permuted) arrays here
    base_vtb_t = torch.from_numpy(base_vtb_np)
    model_out = predict_all(model, base_vtb_t, tensors, test_idx)
    pred = base_vtb_t[test_idx].numpy() + model_out
    err, n = masked_error_at_step(pred, truth_np[test_idx], mask_np[test_idx], PRIMARY_STEP)

    base_err, base_n = masked_error_at_step(
        base_vtb_np[test_idx], truth_np[test_idx], mask_np[test_idx], PRIMARY_STEP)

    print("\n" + "=" * 70)
    print("Check B result")
    print("=" * 70)
    print(f"permuted-target model, out-of-fold (real test set) @1.9s: {err:.4f}  (n={n})")
    print(f"velocity_turning_to_ball baseline, same test set @1.9s:   {base_err:.4f}")
    print(f"real (unpermuted) K2 fold0 result, for reference:         1.8205")
    print(f"\npermuted model beats the baseline: {err < base_err}")


if __name__ == "__main__":
    main()
