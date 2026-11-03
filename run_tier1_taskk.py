"""Tier 1 Task K: train and evaluate the three residual/direct model
variants (TIER1_SPEC_6.md, amending TIER1_SPEC_4.md's Task K).

    python run_tier1_taskk.py

K1 -- residual, robust base:   tbl_kalman_speed(sample, h) + model(sample)
K2 -- residual, clean champion: velocity_turning_to_ball(sample, h) + model(sample)
K3 -- direct:                   model(sample)

Same architecture (model.GRUResidualModel), hyperparameters, folds
(outputs/folds.json), per-fold seed, features, and loss across all
three -- only what the network's output is added to differs. For K3
there is no physics base at all; internally it is implemented as
"add the network output to the anchor position, held constant across
the horizon" (mathematically identical to baselines.constant_position),
purely so the same training code path covers all three variants without
a branch -- K3's number is exactly model(sample) in absolute field
coordinates, nothing else contributes to it.

Uses outputs/folds.json (Task P) unchanged -- no other partition, no
per-variant tuning, test split untouched (Task L only).
"""

from __future__ import annotations

import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from cache_pipeline import build_dataset_cached
from config import MAX_HORIZON, OUTPUT_DIR, SPLIT_SEED, WEEKS
from model import (HIST_DIM, STATIC_DIM, Example, GRUResidualModel,
                  build_examples, check_base_is_absolute, masked_mde_loss)
from run_tier1_taskp import build_predictors_quadrature
from kalman import calibrate_sigma_a, estimate_sigma_a

PRIMARY_STEP = 19   # 1.9s
STEP_3S = 30
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 15
LR = 1e-3
DEVICE = torch.device("cpu")   # see report: measured against mps, cpu wins at this size
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


def save_checkpoint(model: GRUResidualModel, opt: Adam, variant: str, fold_id: int,
                    seed: int, best_val: float, epochs_run: int) -> None:
    """Every train_one call writes one of these before returning -- an
    investigation that needs a trained model (e.g. an alignment check on
    its predictions) should never again require a fresh retrain to get
    one. Saved on top of the *early-stopped* (best-val) weights, which is
    what train_one/predict_all actually hand back and score; the
    optimizer state is whatever it was at the point training stopped
    (patience exhausted or MAX_EPOCHS reached), not rewound to match --
    resuming training from this checkpoint would replay a few
    already-unproductive epochs, not corrupt anything.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{variant}_fold{fold_id}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "seed": seed,
        "fold_id": fold_id,
        "variant": variant,
        "val_loss": best_val,
        "epochs_run": epochs_run,
        "config": {
            "hidden": model.hidden,
            "hist_dim": HIST_DIM,
            "static_dim": STATIC_DIM,
            "max_horizon": MAX_HORIZON,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "lr": LR,
        },
    }, path)
    print(f"  checkpoint written: {path}")


# ------------------------------------------------------------ tensors

def build_tensors(examples: list[Example]):
    n = len(examples)
    global_max_hist = max(e.hist_feat.shape[0] for e in examples)

    hist_padded = np.zeros((n, global_max_hist, examples[0].hist_feat.shape[1]), dtype=np.float32)
    hist_lengths = np.zeros(n, dtype=np.int64)
    static_feat = np.zeros((n, examples[0].static_feat.shape[0]), dtype=np.float32)
    anchor = np.zeros((n, 2), dtype=np.float32)
    truth_padded = np.zeros((n, MAX_HORIZON, 2), dtype=np.float32)
    mask = np.zeros((n, MAX_HORIZON), dtype=np.float32)
    horizon = np.zeros(n, dtype=np.int64)
    game_id = np.zeros(n, dtype=np.int64)

    for i, e in enumerate(examples):
        L = e.hist_feat.shape[0]
        hist_padded[i, :L] = e.hist_feat
        hist_lengths[i] = L
        static_feat[i] = e.static_feat
        anchor[i] = e.anchor
        h = min(e.horizon, MAX_HORIZON)
        truth_padded[i, :h] = e.truth[:h]
        mask[i, :h] = 1.0
        horizon[i] = e.horizon
        game_id[i] = e.game_id

    return {
        "hist_padded": torch.from_numpy(hist_padded),
        "hist_lengths": torch.from_numpy(hist_lengths),
        "static_feat": torch.from_numpy(static_feat),
        "anchor": torch.from_numpy(anchor),
        "truth": torch.from_numpy(truth_padded),
        "mask": torch.from_numpy(mask),
        "horizon": horizon,
        "game_id": game_id,
    }


def compute_base_arrays(samples: list, base_fn, examples: list[Example]) -> np.ndarray:
    """base_fn(sample, MAX_HORIZON) for every sample, flattened in the
    same (sample, row-in-state) order build_examples used -- both walk
    `samples` and each sample's `state` rows in the same order, off the
    same underlying DataFrame, so index i here lines up with
    examples[i] without needing an explicit join key.
    """
    out = np.zeros((len(examples), MAX_HORIZON, 2), dtype=np.float32)
    pos = 0
    for sample in samples:
        n = len(sample.state)
        pred = base_fn(sample, MAX_HORIZON)   # (n, MAX_HORIZON, 2), absolute field coords
        out[pos:pos + n] = pred
        pos += n
    assert pos == len(examples)
    return out


# ------------------------------------------------------------ training

def train_one(variant: str, base: torch.Tensor, tensors: dict,
              train_idx: np.ndarray, val_idx: np.ndarray, seed: int,
              fold_id: int | None = None, checkpoint: bool = True):
    torch.manual_seed(seed)
    model = GRUResidualModel().to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)

    def run_split(idx: np.ndarray, train: bool):
        if train:
            model.train()
        else:
            model.eval()
        rng = np.random.default_rng(seed) if train else None
        order = rng.permutation(idx) if train else idx
        total_loss, total_n = 0.0, 0
        for start in range(0, len(order), BATCH_SIZE):
            b = order[start:start + BATCH_SIZE]
            hist = tensors["hist_padded"][b].to(DEVICE)
            lengths = tensors["hist_lengths"][b]
            static = tensors["static_feat"][b].to(DEVICE)
            truth = tensors["truth"][b].to(DEVICE)
            mask = tensors["mask"][b].to(DEVICE)
            b_base = base[b].to(DEVICE)

            if train:
                opt.zero_grad()
                model_out = model(hist, lengths, static)
                pred = b_base + model_out
                loss = masked_mde_loss(pred, truth, mask)
                loss.backward()
                opt.step()
            else:
                with torch.no_grad():
                    model_out = model(hist, lengths, static)
                    pred = b_base + model_out
                    loss = masked_mde_loss(pred, truth, mask)
            total_loss += loss.item() * len(b)
            total_n += len(b)
        return total_loss / total_n

    best_val = float("inf")
    best_state = None
    patience_left = PATIENCE
    t0 = time.time()
    epochs_run = 0
    for epoch in range(MAX_EPOCHS):
        epochs_run = epoch + 1
        run_split(train_idx, train=True)
        val_loss = run_split(val_idx, train=False)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    wall_clock = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    if checkpoint and fold_id is not None:
        save_checkpoint(model, opt, variant, fold_id, seed, best_val, epochs_run)
    return model, best_val, wall_clock, epochs_run


def predict_all(model: GRUResidualModel, base: torch.Tensor, tensors: dict,
                idx: np.ndarray):
    model.eval()
    with torch.no_grad():
        hist = tensors["hist_padded"][idx].to(DEVICE)
        lengths = tensors["hist_lengths"][idx]
        static = tensors["static_feat"][idx].to(DEVICE)
        model_out = model(hist, lengths, static).cpu().numpy()
    return model_out   # raw network output, (len(idx), MAX_HORIZON, 2)


# ------------------------------------------------------------ scoring

def masked_error_at_step(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray,
                         step: int) -> tuple[float, int]:
    """mean error, n at horizon step `step` (1-indexed like evaluate.py)."""
    col = step - 1
    valid = mask[:, col] > 0
    if valid.sum() == 0:
        return float("nan"), 0
    err = np.linalg.norm(pred[valid, col] - truth[valid, col], axis=-1)
    return float(err.mean()), int(valid.sum())


def main():
    print(f"loading {len(WEEKS)} weeks via cache_pipeline...")
    samples, _ = build_dataset_cached(WEEKS)

    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    assert fold_info["seed"] == SPLIT_SEED
    print(f"{len(folds)} folds loaded from outputs/folds.json")

    train_val_games = set(g for fold in folds for g in fold)
    train_val_samples = [s for s in samples if s.game_id in train_val_games]
    print(f"{len(train_val_samples):,} train+val plays")

    examples = build_examples(train_val_samples)
    print(f"{len(examples):,} train+val tracks (examples), "
         f"anchor-frame relative-position check passed")

    tensors = build_tensors(examples)
    n_total = len(examples)
    n_params = GRUResidualModel().count_parameters()
    print(f"model parameters: {n_params:,} vs {n_total:,} training-eligible "
         f"tracks (train+val total; each fold trains on ~3/5 of this)")

    # -------- frozen physics bases, computed once on train+val, exactly
    # as Task P did (same sigma_a, same quadrature convention, no
    # set_filter_R call) -- these are fixed inputs to K1/K2, not refit
    # per fold; baselines have no training step.
    sigma_a_raw = estimate_sigma_a(train_val_samples)
    sigma_a = calibrate_sigma_a(train_val_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    print(f"sigma_a = {sigma_a:.4f} (train+val only, matches Task P)")
    predictors = build_predictors_quadrature(sigma_a)

    base_tks = compute_base_arrays(train_val_samples, predictors["tbl_kalman_speed"], examples)
    base_vtb = compute_base_arrays(train_val_samples, predictors["velocity_turning_to_ball"], examples)
    check_base_is_absolute(base_tks, "tbl_kalman_speed")
    check_base_is_absolute(base_vtb, "velocity_turning_to_ball")

    anchor_np = tensors["anchor"].numpy()
    base_anchor_repeat = np.repeat(anchor_np[:, None, :], MAX_HORIZON, axis=1)

    bases = {
        "K1": torch.from_numpy(base_tks),
        "K2": torch.from_numpy(base_vtb),
        "K3": torch.from_numpy(base_anchor_repeat),
    }

    game_id_arr = tensors["game_id"]
    truth_np = tensors["truth"].numpy()
    mask_np = tensors["mask"].numpy()

    n_folds = len(folds)
    per_fold_scores = {v: [] for v in ("K1", "K2", "K3")}
    per_fold_scores_3s = {v: [] for v in ("K1", "K2", "K3")}
    wall_clocks = {v: [] for v in ("K1", "K2", "K3")}
    epochs_used = {v: [] for v in ("K1", "K2", "K3")}

    # out-of-fold storage, aligned to `examples` order, filled once per track
    oof_model_out = {v: np.full((n_total, MAX_HORIZON, 2), np.nan, dtype=np.float32)
                     for v in ("K1", "K2", "K3")}
    oof_pred = {v: np.full((n_total, MAX_HORIZON, 2), np.nan, dtype=np.float32)
               for v in ("K1", "K2", "K3")}
    oof_filled = np.zeros(n_total, dtype=bool)

    for i in range(n_folds):
        test_games = set(folds[i])
        val_games = set(folds[(i + 1) % n_folds])
        train_games = train_val_games - test_games - val_games

        test_idx = np.where(np.isin(game_id_arr, list(test_games)))[0]
        val_idx = np.where(np.isin(game_id_arr, list(val_games)))[0]
        train_idx = np.where(np.isin(game_id_arr, list(train_games)))[0]
        assert len(test_idx) + len(val_idx) + len(train_idx) == n_total

        seed = SPLIT_SEED + i
        print(f"\n=== fold {i} (test={len(test_games)} games/{len(test_idx)} tracks, "
             f"val={len(val_games)} games/{len(val_idx)} tracks, "
             f"train={len(train_games)} games/{len(train_idx)} tracks) ===")

        for variant in ("K1", "K2", "K3"):
            base = bases[variant]
            model, best_val, wall_clock, epochs_run = train_one(
                variant, base, tensors, train_idx, val_idx, seed, fold_id=i)
            wall_clocks[variant].append(wall_clock)
            epochs_used[variant].append(epochs_run)

            model_out = predict_all(model, base, tensors, test_idx)
            pred = base[test_idx].numpy() + model_out

            oof_model_out[variant][test_idx] = model_out
            oof_pred[variant][test_idx] = pred
            oof_filled[test_idx] = True

            err_primary, n_primary = masked_error_at_step(
                pred, truth_np[test_idx], mask_np[test_idx], PRIMARY_STEP)
            err_3s, n_3s = masked_error_at_step(
                pred, truth_np[test_idx], mask_np[test_idx], STEP_3S)
            per_fold_scores[variant].append(err_primary)
            per_fold_scores_3s[variant].append(err_3s)

            print(f"  {variant}: val_loss={best_val:.4f} epochs={epochs_run} "
                 f"wall_clock={wall_clock:.1f}s "
                 f"@primary={err_primary:.4f} (n={n_primary}) "
                 f"@3s={err_3s:.4f} (n={n_3s})")

    assert oof_filled.all(), "every track must get exactly one out-of-fold prediction"

    print("\n" + "=" * 70)
    print("Task K -- CV summary (5-fold, out-of-fold, quadrature convention)")
    print("=" * 70)
    rows = []
    for variant in ("K1", "K2", "K3"):
        scores = np.array(per_fold_scores[variant])
        scores_3s = np.array(per_fold_scores_3s[variant])
        rows.append({
            "variant": variant,
            "cv_mean_primary": scores.mean(),
            "cv_std_primary": scores.std(ddof=1),
            "cv_min_primary": scores.min(),
            "cv_max_primary": scores.max(),
            "cv_mean_3s": scores_3s.mean(),
            "cv_std_3s": scores_3s.std(ddof=1),
            "mean_wall_clock_s": np.mean(wall_clocks[variant]),
            "mean_epochs": np.mean(epochs_used[variant]),
        })
    summary = pd.DataFrame(rows).round(4)
    print(summary.to_string(index=False))
    summary.to_csv(OUTPUT_DIR / "tier1_taskk_cv_summary.csv", index=False)

    # -------- (a) K1-vs-K2 final-prediction distance vs base distance
    print("\n" + "=" * 70)
    print("K1 vs K2: final-prediction distance, out-of-fold, at primary horizon")
    print("=" * 70)
    col = PRIMARY_STEP - 1
    valid = (mask_np[:, col] > 0)
    pred_dist = np.linalg.norm(oof_pred["K1"][valid, col] - oof_pred["K2"][valid, col], axis=-1)
    base_dist = np.linalg.norm(base_tks[valid, col] - base_vtb[valid, col], axis=-1)
    ratio = pred_dist.mean() / base_dist.mean()
    print(f"  mean |K1_pred - K2_pred| = {pred_dist.mean():.4f} yd (n={valid.sum()})")
    print(f"  mean |base_tks - base_vtb| = {base_dist.mean():.4f} yd")
    print(f"  ratio (pred distance / base distance) = {ratio:.4f}")

    # -------- (b) residual magnitude, K1 and K2, over unmasked steps
    print("\n" + "=" * 70)
    print("Residual magnitude ||model_out|| (raw network output), K1 vs K2")
    print("=" * 70)
    report_steps = [5, 10, PRIMARY_STEP, STEP_3S]
    for variant in ("K1", "K2"):
        print(f"\n  {variant}:")
        for step in report_steps:
            c = step - 1
            v = mask_np[:, c] > 0
            if v.sum() == 0:
                continue
            mag = np.linalg.norm(oof_model_out[variant][v, c], axis=-1)
            print(f"    step {step:2d} ({step/10:.1f}s, n={v.sum():5d}): "
                 f"mean={mag.mean():.4f} median={np.median(mag):.4f} "
                 f"p90={np.percentile(mag, 90):.4f}")

    # bias-correction vs genuine-learning check, at primary horizon
    c = PRIMARY_STEP - 1
    v = mask_np[:, c] > 0
    k1_res = oof_model_out["K1"][v, c]
    k2_res = oof_model_out["K2"][v, c]
    k1_mag = np.linalg.norm(k1_res, axis=-1)
    k2_mag = np.linalg.norm(k2_res, axis=-1)
    diff = k1_res - k2_res
    diff_mag = np.linalg.norm(diff, axis=-1)
    print(f"\n  K1 residual mean magnitude:  {k1_mag.mean():.4f} yd")
    print(f"  K2 residual mean magnitude:  {k2_mag.mean():.4f} yd")
    print(f"  |K1_residual - K2_residual| mean: {diff_mag.mean():.4f} yd, "
         f"std: {diff_mag.std():.4f} yd")
    print(f"  (base gap tbl_kalman_speed vs velocity_turning_to_ball on "
         f"this fold data: 1.03 yd per Task P/user correction)")

    print("\nTask K complete. Stopping as instructed -- no Task L or M yet.")


if __name__ == "__main__":
    main()
