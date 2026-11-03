"""K5: K3's architecture and features (anchor-only base, finite-
differenced velocity), trained WITH on-the-fly noise augmentation at
sigma_player=0.2 (derived-mode semantics: position noise shared with
the anchor subtraction, velocity noise from finite-differencing that
same noise realisation -- not independent noise injected straight into
vx,vy, which would misrepresent how real degradation actually
correlates position and velocity error). Fresh noise redrawn every
batch, every epoch -- true augmentation, not one fixed noisy copy.

Validation (early stopping) stays on CLEAN data, so the only
intervention is training-time exposure to noise -- "everything else
identical" per the brief. Same architecture, folds, per-fold seed
convention, checkpointing as K1-K4.
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
from config import DT, OUTPUT_DIR, SPLIT_SEED, WEEKS
from model import GRUResidualModel, build_examples, masked_mde_loss
from run_tier1_taskk import (BATCH_SIZE, DEVICE, MAX_EPOCHS, PATIENCE, LR,
                             build_tensors, masked_error_at_step, predict_all,
                             save_checkpoint)

PRIMARY_STEP = 19
STEP_3S = 30
AUGMENT_SIGMA = 0.2


def augment_hist_batch(hist: torch.Tensor, lengths: torch.Tensor, sigma: float) -> torch.Tensor:
    """Fresh position-noise draw per call, propagated into rel_x/rel_y
    (anchor-relative, so every track shares its own anchor's noise term)
    and into vx/vy via finite-differencing that SAME noise realisation
    -- matching degrade.py's actual mechanism, not independent per-
    channel noise.
    """
    B, T, _ = hist.shape
    device = hist.device
    eps_x = torch.randn(B, T, device=device) * sigma
    eps_y = torch.randn(B, T, device=device) * sigma
    valid = (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1).to(device))
    eps_x = eps_x * valid
    eps_y = eps_y * valid

    anchor_idx = (lengths - 1).clamp(min=0).to(device)
    br = torch.arange(B, device=device)
    eps_x_anchor = eps_x[br, anchor_idx].unsqueeze(1)
    eps_y_anchor = eps_y[br, anchor_idx].unsqueeze(1)

    dx = eps_x - torch.roll(eps_x, shifts=1, dims=1)
    dy = eps_y - torch.roll(eps_y, shifts=1, dims=1)
    dx[:, 0] = dx[:, 1] if T > 1 else 0.0
    dy[:, 0] = dy[:, 1] if T > 1 else 0.0

    out = hist.clone()
    out[:, :, 0] = hist[:, :, 0] + eps_x - eps_x_anchor
    out[:, :, 1] = hist[:, :, 1] + eps_y - eps_y_anchor
    out[:, :, 2] = hist[:, :, 2] + dx / DT
    out[:, :, 3] = hist[:, :, 3] + dy / DT
    return out


def train_one_k5(base: torch.Tensor, tensors: dict, train_idx: np.ndarray,
                 val_idx: np.ndarray, seed: int, fold_id: int):
    torch.manual_seed(seed)
    model = GRUResidualModel().to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)

    def run_split(idx, train):
        model.train() if train else model.eval()
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
                hist = augment_hist_batch(hist, lengths, AUGMENT_SIGMA)
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
    save_checkpoint(model, opt, "K5", fold_id, seed, best_val, epochs_run)
    return model, best_val, wall_clock, epochs_run


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
    anchor_np = tensors["anchor"].numpy()
    base_anchor_repeat = np.repeat(anchor_np[:, None, :], truth_np.shape[1], axis=1)
    base_t = torch.from_numpy(base_anchor_repeat)

    n_folds = len(folds)
    per_fold_scores, per_fold_scores_3s = [], []
    for i in range(n_folds):
        test_games = set(folds[i])
        val_games = set(folds[(i + 1) % n_folds])
        train_games = train_val_games - test_games - val_games
        test_idx = np.where(np.isin(game_id_arr, list(test_games)))[0]
        val_idx = np.where(np.isin(game_id_arr, list(val_games)))[0]
        train_idx = np.where(np.isin(game_id_arr, list(train_games)))[0]
        seed = SPLIT_SEED + i

        model, best_val, wall_clock, epochs_run = train_one_k5(
            base_t, tensors, train_idx, val_idx, seed, fold_id=i)
        model_out = predict_all(model, base_t, tensors, test_idx)
        pred = base_anchor_repeat[test_idx] + model_out
        err_p, n_p = masked_error_at_step(pred, truth_np[test_idx], mask_np[test_idx], PRIMARY_STEP)
        err_3, n_3 = masked_error_at_step(pred, truth_np[test_idx], mask_np[test_idx], STEP_3S)
        per_fold_scores.append(err_p)
        per_fold_scores_3s.append(err_3)
        print(f"fold {i}: epochs={epochs_run} wall_clock={wall_clock:.1f}s "
             f"@1.9s={err_p:.4f} (n={n_p}) @3s={err_3:.4f} (n={n_3})")

    s = np.array(per_fold_scores)
    s3 = np.array(per_fold_scores_3s)
    print(f"\nK5 CV: mean_1.9s={s.mean():.4f} std={s.std(ddof=1):.4f} "
         f"min={s.min():.4f} max={s.max():.4f}")
    print(f"K5 CV: mean_3s={s3.mean():.4f} std={s3.std(ddof=1):.4f}")

    row = pd.DataFrame([{
        "predictor": "K5", "cv_mean_1.9s": s.mean(), "cv_std_1.9s": s.std(ddof=1),
        "cv_min_1.9s": s.min(), "cv_max_1.9s": s.max(),
        "cv_mean_3s": s3.mean(), "cv_std_3s": s3.std(ddof=1),
    }]).round(4)
    cv_path = OUTPUT_DIR / "tier1_taskl_cv_summary.csv"
    existing = pd.read_csv(cv_path)
    existing = existing[existing["predictor"] != "K5"]
    combined = pd.concat([existing, row], ignore_index=True)
    combined.to_csv(cv_path, index=False)
    print("\nK1-K5 CV summary, combined:")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
