"""The test-split pass. TIER1_SPEC_6.md / agreed conditions:

1. Early-stopping slice split BY GAME from the 231 train+val games,
   fixed by seed, written to outputs/final_val_games.json.
2. Every variant (K1-K5) and every baseline scored in the SAME single
   pass over the test split.
3. Baselines recomputed on the identical test plays, per-role reported.
4. Final models are trained on all 231 train+val games (minus the
   early-stopping slice); the CV models saw 4/5 of them. The test
   number and the CV number are NOT directly comparable -- test is the
   headline, CV fold spread is the uncertainty estimate.

Test split (outputs/splits.json's "test" games) is read here for the
first time in this entire project. One pass. No repeats.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from baselines import BASELINES
from cache_pipeline import build_dataset_cached
from config import MAX_HORIZON, OUTPUT_DIR, SPLIT_SEED, WEEKS
from kalman import build_Q, calibrate_sigma_a, estimate_sigma_a
from model import (GRUResidualModel, build_examples, build_examples_k4,
                  masked_mde_loss)
from run_tier1_k5 import augment_hist_batch, AUGMENT_SIGMA
from run_tier1_taskk import (BATCH_SIZE, DEVICE, LR, MAX_EPOCHS, PATIENCE,
                             build_tensors, compute_base_arrays,
                             masked_error_at_step, predict_all,
                             save_checkpoint)
from run_tier1_taskp import build_predictors_quadrature

PRIMARY_STEP = 19
STEP_3S = 30
FINAL_VAL_FRACTION = 0.20
FINAL_SEED = SPLIT_SEED

ALL_BASELINE_NAMES = [
    "constant_position", "constant_velocity", "constant_acceleration",
    "damped_velocity", "toward_ball_land", "velocity_turning_to_ball",
    "kalman", "kalman_velocity", "kalman_ca", "imm",
    "tbl_kalman_speed", "tbl_kalman_pos", "tbl_kalman_both",
]


def masked_error_by_role(pred, truth, mask, roles, step):
    col = step - 1
    valid = mask[:, col] > 0
    if valid.sum() == 0:
        return {}
    err = np.linalg.norm(pred[valid, col] - truth[valid, col], axis=-1)
    r = roles[valid]
    out = {}
    for role in np.unique(r):
        m = r == role
        out[role] = (float(err[m].mean()), int(m.sum()))
    out["overall"] = (float(err.mean()), int(valid.sum()))
    return out


def train_final(variant, base, tensors, train_idx, val_idx, seed, augment=False):
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
                if augment:
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

    import copy, time
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
    save_checkpoint(model, opt, f"{variant}_FINAL", 0, seed, best_val, epochs_run)
    print(f"  {variant} final: epochs={epochs_run} wall_clock={wall_clock:.1f}s val_loss={best_val:.4f}")
    return model


def main():
    samples, _ = build_dataset_cached(WEEKS)
    with open(OUTPUT_DIR / "splits.json") as f:
        splits = json.load(f)
    train_val_games = splits["train"] + splits["val"]
    test_games = set(splits["test"])

    # -------- condition 1: early-stopping slice, by game, fixed seed, written to disk
    rng = np.random.default_rng(FINAL_SEED)
    shuffled = rng.permutation(sorted(train_val_games))
    n_val = int(round(FINAL_VAL_FRACTION * len(shuffled)))
    final_val_games = sorted(int(g) for g in shuffled[:n_val])
    final_train_games = sorted(int(g) for g in shuffled[n_val:])
    with open(OUTPUT_DIR / "final_val_games.json", "w") as f:
        json.dump({"seed": FINAL_SEED, "fraction": FINAL_VAL_FRACTION,
                  "final_train_games": final_train_games,
                  "final_val_games": final_val_games}, f, indent=2)
    print(f"final train games: {len(final_train_games)}, final val (early-stop) games: "
         f"{len(final_val_games)}, test games: {len(test_games)} (untouched until now)")

    train_val_samples = [s for s in samples if s.game_id in set(train_val_games)]
    test_samples = [s for s in samples if s.game_id in test_games]

    # -------- standard features (K1, K2, K3, K5)
    examples_std = build_examples(train_val_samples)
    tensors_std = build_tensors(examples_std)
    game_id_std = tensors_std["game_id"]
    final_train_idx = np.where(np.isin(game_id_std, final_train_games))[0]
    final_val_idx = np.where(np.isin(game_id_std, final_val_games))[0]

    sigma_a_raw = estimate_sigma_a(train_val_samples)
    sigma_a = calibrate_sigma_a(train_val_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    Q = build_Q(sigma_a)
    predictors = build_predictors_quadrature(sigma_a)

    base_tks_std = compute_base_arrays(train_val_samples, predictors["tbl_kalman_speed"], examples_std)
    base_vtb_std = compute_base_arrays(train_val_samples, predictors["velocity_turning_to_ball"], examples_std)
    anchor_std = tensors_std["anchor"].numpy()
    base_anchor_std = np.repeat(anchor_std[:, None, :], tensors_std["truth"].shape[1], axis=1)

    print("\ntraining final models (all 231 train+val games minus early-stop slice)...")
    model_k1 = train_final("K1", torch.from_numpy(base_tks_std), tensors_std, final_train_idx, final_val_idx, FINAL_SEED)
    model_k2 = train_final("K2", torch.from_numpy(base_vtb_std), tensors_std, final_train_idx, final_val_idx, FINAL_SEED)
    model_k3 = train_final("K3", torch.from_numpy(base_anchor_std), tensors_std, final_train_idx, final_val_idx, FINAL_SEED)
    model_k5 = train_final("K5", torch.from_numpy(base_anchor_std), tensors_std, final_train_idx, final_val_idx, FINAL_SEED, augment=True)

    # -------- K4 features (Kalman-filtered velocity)
    examples_k4 = build_examples_k4(train_val_samples, Q)
    tensors_k4 = build_tensors(examples_k4)
    game_id_k4 = tensors_k4["game_id"]
    final_train_idx_k4 = np.where(np.isin(game_id_k4, final_train_games))[0]
    final_val_idx_k4 = np.where(np.isin(game_id_k4, final_val_games))[0]
    anchor_k4 = tensors_k4["anchor"].numpy()
    base_anchor_k4 = np.repeat(anchor_k4[:, None, :], tensors_k4["truth"].shape[1], axis=1)
    model_k4 = train_final("K4", torch.from_numpy(base_anchor_k4), tensors_k4, final_train_idx_k4, final_val_idx_k4, FINAL_SEED)

    # ================================================================
    # condition 2/3: single pass over the TEST split -- models and baselines
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST SPLIT -- first and only look, this run")
    print("=" * 70)

    examples_test_std = build_examples(test_samples)
    tensors_test_std = build_tensors(examples_test_std)
    roles_test_std = np.array([e.role for e in examples_test_std])
    truth_test_std = tensors_test_std["truth"].numpy()
    mask_test_std = tensors_test_std["mask"].numpy()
    test_idx_all_std = np.arange(len(examples_test_std))

    examples_test_k4 = build_examples_k4(test_samples, Q)
    tensors_test_k4 = build_tensors(examples_test_k4)
    roles_test_k4 = np.array([e.role for e in examples_test_k4])
    truth_test_k4 = tensors_test_k4["truth"].numpy()
    mask_test_k4 = tensors_test_k4["mask"].numpy()
    test_idx_all_k4 = np.arange(len(examples_test_k4))

    results = []

    # -------- baselines, recomputed on the test plays, same quadrature convention (clean, sigma=0)
    for name in ALL_BASELINE_NAMES:
        arr = compute_base_arrays(test_samples, predictors[name], examples_test_std)
        for step, label in ((PRIMARY_STEP, "primary"), (STEP_3S, "3s")):
            by_role = masked_error_by_role(arr, truth_test_std, mask_test_std, roles_test_std, step)
            for role, (err, n) in by_role.items():
                results.append({"predictor": name, "kind": "baseline", "horizon": label,
                               "role": role, "test_error": err, "n": n})

    # -------- K1/K2/K3/K5 (standard features)
    anchor_test_std = tensors_test_std["anchor"].numpy()
    base_anchor_test_std = np.repeat(anchor_test_std[:, None, :], tensors_test_std["truth"].shape[1], axis=1)
    base_tks_test = compute_base_arrays(test_samples, predictors["tbl_kalman_speed"], examples_test_std)
    base_vtb_test = compute_base_arrays(test_samples, predictors["velocity_turning_to_ball"], examples_test_std)

    for variant, model, base_test in (
        ("K1", model_k1, base_tks_test), ("K2", model_k2, base_vtb_test),
        ("K3", model_k3, base_anchor_test_std), ("K5", model_k5, base_anchor_test_std),
    ):
        base_t = torch.from_numpy(base_test)
        model_out = predict_all(model, base_t, tensors_test_std, test_idx_all_std)
        pred = base_test[test_idx_all_std] + model_out
        for step, label in ((PRIMARY_STEP, "primary"), (STEP_3S, "3s")):
            by_role = masked_error_by_role(pred, truth_test_std, mask_test_std, roles_test_std, step)
            for role, (err, n) in by_role.items():
                results.append({"predictor": variant, "kind": "model", "horizon": label,
                               "role": role, "test_error": err, "n": n})

    # -------- K4 (Kalman-velocity features)
    anchor_test_k4 = tensors_test_k4["anchor"].numpy()
    base_anchor_test_k4 = np.repeat(anchor_test_k4[:, None, :], tensors_test_k4["truth"].shape[1], axis=1)
    base_t4 = torch.from_numpy(base_anchor_test_k4)
    model_out4 = predict_all(model_k4, base_t4, tensors_test_k4, test_idx_all_k4)
    pred4 = base_anchor_test_k4[test_idx_all_k4] + model_out4
    for step, label in ((PRIMARY_STEP, "primary"), (STEP_3S, "3s")):
        by_role = masked_error_by_role(pred4, truth_test_k4, mask_test_k4, roles_test_k4, step)
        for role, (err, n) in by_role.items():
            results.append({"predictor": "K4", "kind": "model", "horizon": label,
                           "role": role, "test_error": err, "n": n})

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "tier1_test_pass_full.csv", index=False)

    # -------- combine with CV numbers into the final headline table
    cv = pd.read_csv(OUTPUT_DIR / "tier1_taskl_cv_summary.csv")
    overall_primary = results_df[(results_df["horizon"] == "primary") & (results_df["role"] == "overall")]
    overall_3s = results_df[(results_df["horizon"] == "3s") & (results_df["role"] == "overall")]

    final_rows = []
    for _, row in cv.iterrows():
        pname = row["predictor"]
        t_row = overall_primary[overall_primary["predictor"] == pname]
        t3_row = overall_3s[overall_3s["predictor"] == pname]
        final_rows.append({
            "predictor": pname, "cv_mean_1.9s": row["cv_mean_1.9s"], "cv_std_1.9s": row["cv_std_1.9s"],
            "test_1.9s": t_row["test_error"].iloc[0] if len(t_row) else np.nan,
            "test_1.9s_n": t_row["n"].iloc[0] if len(t_row) else np.nan,
            "test_3s": t3_row["test_error"].iloc[0] if len(t3_row) else np.nan,
            "test_3s_n": t3_row["n"].iloc[0] if len(t3_row) else np.nan,
        })
    for name in ALL_BASELINE_NAMES:
        t_row = overall_primary[overall_primary["predictor"] == name]
        t3_row = overall_3s[overall_3s["predictor"] == name]
        final_rows.append({
            "predictor": name, "cv_mean_1.9s": np.nan, "cv_std_1.9s": np.nan,
            "test_1.9s": t_row["test_error"].iloc[0] if len(t_row) else np.nan,
            "test_1.9s_n": t_row["n"].iloc[0] if len(t_row) else np.nan,
            "test_3s": t3_row["test_error"].iloc[0] if len(t3_row) else np.nan,
            "test_3s_n": t3_row["n"].iloc[0] if len(t3_row) else np.nan,
        })
    final_df = pd.DataFrame(final_rows).round(4).sort_values("test_1.9s")
    final_df.to_csv(OUTPUT_DIR / "tier1_test_pass_headline.csv", index=False)

    print("\n" + "=" * 70)
    print("HEADLINE TABLE -- CV mean/spread vs test @1.9s and @3s, sorted by test@1.9s")
    print("=" * 70)
    print(final_df.to_string(index=False))

    print("\nper-role, test split, @1.9s:")
    role_view = results_df[(results_df["horizon"] == "primary")].pivot_table(
        index="predictor", columns="role", values="test_error")
    print(role_view.round(4).to_string())

    print("\nNOTE: final models were trained on all 231 train+val games "
         "(minus a 20% early-stopping slice); the CV models above each "
         "saw only 4/5 of those 231 games. The CV and test numbers are "
         "NOT directly comparable -- test is the headline, CV fold "
         "spread is the uncertainty estimate.")


if __name__ == "__main__":
    main()
