"""Item 2: degradation sweep on the TEST split, using the FINAL models
(trained once, frozen before the first test look) -- not the CV fold
models. Measures whether the CV-based K5/tbl_kalman_speed crossing
(1.055) replicates on genuinely held-out games, or whether it was
optimistic.

This is a measurement on already-frozen models, not a second selection
step: nothing here is tuned, no architecture/hyperparameter/variant
decision depends on this sweep's outcome. The test split was already
spent once (run_tier1_test_pass.py); this is a further look at the
SAME frozen artifacts under noise, not a new choice among them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from baselines import BASELINES
from config import OUTPUT_DIR, WEEKS
from data_loader import add_kinematics, build_samples, load_raw, normalise
from degrade import corrupt_input
from kalman import build_Q, calibrate_sigma_a, estimate_sigma_a
from model import GRUResidualModel, build_examples, build_examples_k4
from run_tier1_crossover import find_crossing
import run_tier1_crossover as co
from run_tier1_taskk import build_tensors, masked_error_at_step, predict_all
from run_tier1_taskp import build_predictors_quadrature
from run_tier1_r_convention_check import set_filter_R_convention

PRIMARY_STEP = 19
VELOCITY_SOURCE = "derived"
DEGRADE_BASE_SEED_M = 20260929
# coarse coverage + fine resolution around the CV crossing (~1.055)
SIGMA_GRID = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 5.0]
FINE_GRID = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
SEEDS = [0, 1, 2, 3, 4]   # more seeds than the CV sweep -- test has far fewer plays (41 games)


def load_final(variant):
    ckpt = torch.load(OUTPUT_DIR / "checkpoints" / f"{variant}_FINAL_fold0.pt", weights_only=False)
    model = GRUResidualModel(hidden=ckpt["config"]["hidden"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    with open(OUTPUT_DIR / "splits.json") as f:
        splits = json.load(f)
    test_games = set(splits["test"])
    train_val_games = set(splits["train"] + splits["val"])

    print("loading raw input/output, test games only...")
    inp, out = load_raw(weeks=WEEKS)
    inp = inp[inp["game_id"].isin(test_games)].reset_index(drop=True)
    out = out[out["game_id"].isin(test_games)].reset_index(drop=True)
    inp, out = normalise(inp, out)
    clean_test_inp = add_kinematics(inp)
    clean_test_out = out
    print(f"  {len(clean_test_inp):,} input rows, {len(clean_test_out):,} output rows (test games only)")

    # frozen sigma_a from train+val only -- never re-estimated on test
    print("loading raw input/output, train+val games only, for frozen sigma_a...")
    inp_tv, out_tv = load_raw(weeks=WEEKS)
    inp_tv = inp_tv[inp_tv["game_id"].isin(train_val_games)].reset_index(drop=True)
    out_tv = out_tv[out_tv["game_id"].isin(train_val_games)].reset_index(drop=True)
    inp_tv, out_tv = normalise(inp_tv, out_tv)
    inp_tv = add_kinematics(inp_tv)
    tv_samples_for_sigma = build_samples(inp_tv, out_tv)
    sigma_a_raw = estimate_sigma_a(tv_samples_for_sigma)
    sigma_a = calibrate_sigma_a(tv_samples_for_sigma, sigma_a_hi=sigma_a_raw, target=2.0)
    Q = build_Q(sigma_a)
    print(f"frozen sigma_a = {sigma_a:.4f}")

    models = {v: load_final(v) for v in ("K1", "K2", "K3", "K4", "K5")}

    all_rows = []
    play_rows = []
    combos = [(s, 0) if s == 0.0 else (s, seed) for s in SIGMA_GRID
             for seed in (SEEDS if s > 0 else [0])]
    for k, (sigma, seed) in enumerate(combos, 1):
        print(f"[{k}/{len(combos)}] sigma={sigma} seed={seed}")
        rng = np.random.default_rng(DEGRADE_BASE_SEED_M + seed)
        degraded_inp = corrupt_input(clean_test_inp, sigma, rng, VELOCITY_SOURCE)
        degraded_samples = build_samples(degraded_inp, clean_test_out)

        examples_std = build_examples(degraded_samples)
        tensors_std = build_tensors(examples_std)
        truth_std = tensors_std["truth"].numpy()
        mask_std = tensors_std["mask"].numpy()
        anchor_std = tensors_std["anchor"].numpy()
        base_anchor_std = np.repeat(anchor_std[:, None, :], truth_std.shape[1], axis=1)
        meta_std = [(e.game_id, e.play_id) for e in examples_std]

        examples_k4 = build_examples_k4(degraded_samples, Q)
        tensors_k4 = build_tensors(examples_k4)
        truth_k4 = tensors_k4["truth"].numpy()
        mask_k4 = tensors_k4["mask"].numpy()
        anchor_k4 = tensors_k4["anchor"].numpy()
        base_anchor_k4 = np.repeat(anchor_k4[:, None, :], truth_k4.shape[1], axis=1)

        set_filter_R_convention(sigma, VELOCITY_SOURCE, "quadrature")
        predictors = build_predictors_quadrature(sigma_a)
        base_tks = np.zeros_like(base_anchor_std)
        base_vtb = np.zeros_like(base_anchor_std)
        tks_raw = np.zeros_like(base_anchor_std)
        pos = 0
        for sample in degraded_samples:
            n = len(sample.state)
            base_tks[pos:pos+n] = predictors["tbl_kalman_speed"](sample, truth_std.shape[1])
            base_vtb[pos:pos+n] = predictors["velocity_turning_to_ball"](sample, truth_std.shape[1])
            pos += n
        tks_raw = base_tks.copy()

        idx_all = np.arange(len(examples_std))
        idx_all_k4 = np.arange(len(examples_k4))
        col = PRIMARY_STEP - 1

        for variant, base, tens, truth, mask in (
            ("K1", base_tks, tensors_std, truth_std, mask_std),
            ("K2", base_vtb, tensors_std, truth_std, mask_std),
            ("K3", base_anchor_std, tensors_std, truth_std, mask_std),
            ("K5", base_anchor_std, tensors_std, truth_std, mask_std),
        ):
            out_m = predict_all(models[variant], torch.from_numpy(base), tens, idx_all)
            pred = base[idx_all] + out_m
            err, n_v = masked_error_at_step(pred, truth, mask, PRIMARY_STEP)
            all_rows.append({"sigma": sigma, "seed": seed, "predictor": variant, "error": err, "n": n_v})
            if sigma in FINE_GRID or sigma == 0.0:
                valid = mask[:, col] > 0
                for local_j in np.where(valid)[0]:
                    d = float(np.linalg.norm(pred[local_j, col] - truth[local_j, col]))
                    gid, pid = meta_std[local_j]
                    play_rows.append((sigma, seed, PRIMARY_STEP, gid, pid, variant, d))

        out4 = predict_all(models["K4"], torch.from_numpy(base_anchor_k4), tensors_k4, idx_all_k4)
        pred4 = base_anchor_k4[idx_all_k4] + out4
        err4, n4 = masked_error_at_step(pred4, truth_k4, mask_k4, PRIMARY_STEP)
        all_rows.append({"sigma": sigma, "seed": seed, "predictor": "K4", "error": err4, "n": n4})

        err_tks, n_tks = masked_error_at_step(tks_raw, truth_std, mask_std, PRIMARY_STEP)
        all_rows.append({"sigma": sigma, "seed": seed, "predictor": "tbl_kalman_speed", "error": err_tks, "n": n_tks})
        if sigma in FINE_GRID or sigma == 0.0:
            valid = mask_std[:, col] > 0
            for local_j in np.where(valid)[0]:
                d = float(np.linalg.norm(tks_raw[local_j, col] - truth_std[local_j, col]))
                gid, pid = meta_std[local_j]
                play_rows.append((sigma, seed, PRIMARY_STEP, gid, pid, "tbl_kalman_speed", d))

        for name in ("constant_velocity", "velocity_turning_to_ball"):
            arr = np.zeros_like(base_anchor_std)
            pos = 0
            for sample in degraded_samples:
                n = len(sample.state)
                arr[pos:pos+n] = BASELINES[name](sample, truth_std.shape[1])
                pos += n
            err_b, n_b = masked_error_at_step(arr, truth_std, mask_std, PRIMARY_STEP)
            all_rows.append({"sigma": sigma, "seed": seed, "predictor": name, "error": err_b, "n": n_b})

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "tier1_test_degrade_raw.csv", index=False)
    summary = df.groupby(["sigma", "predictor"])["error"].mean().reset_index()
    pivot = summary.pivot(index="sigma", columns="predictor", values="error").round(4)
    print("\n" + "=" * 70)
    print("TEST-SPLIT degradation sweep, mean error @1.9s vs sigma_player")
    print("=" * 70)
    print(pivot.to_string())
    pivot.to_csv(OUTPUT_DIR / "tier1_test_degrade_summary.csv")

    # -------- crossing, test split, play-level bootstrap
    play_level = pd.DataFrame(play_rows, columns=[
        "sigma", "seed", "step", "game_id", "play_id", "predictor", "mde"])
    play_level.to_csv(OUTPUT_DIR / "tier1_test_crossover_playlevel.csv", index=False)
    per_play = (play_level.groupby(["sigma", "seed", "step", "game_id", "play_id", "predictor"])
               ["mde"].mean().reset_index())

    full_grid = sorted(set([0.0] + FINE_GRID))
    mat_k5, plays = co._play_sigma_matrix(per_play, full_grid, "K5", PRIMARY_STEP)
    mat_tks, _ = co._play_sigma_matrix(per_play, full_grid, "tbl_kalman_speed", PRIMARY_STEP)

    point = find_crossing(full_grid, np.nanmean(mat_k5, axis=0), np.nanmean(mat_tks, axis=0))
    print(f"\nTEST-SPLIT point-estimate K5/tbl_kalman_speed crossing sigma: {point}")

    rng = np.random.default_rng(0)
    n_plays = len(plays)
    print(f"(bootstrap over {n_plays} test plays -- far fewer than the "
         f"{231} train+val games the CV crossing was bootstrapped over)")
    crossings = []
    for _ in range(1000):
        idx = rng.integers(0, n_plays, size=n_plays)
        mean_k5 = np.nanmean(mat_k5[idx], axis=0)
        mean_tks = np.nanmean(mat_tks[idx], axis=0)
        c = find_crossing(full_grid, mean_k5, mean_tks)
        if c is not None:
            crossings.append(c)
    crossings = np.array(crossings)
    print(f"bootstrap (n={len(crossings)}/1000 valid): "
         f"median={np.median(crossings):.4f} "
         f"95% CI=({np.percentile(crossings,2.5):.4f}, {np.percentile(crossings,97.5):.4f})")
    print(f"K5 mean mde by sigma: {np.round(np.nanmean(mat_k5, axis=0), 4)}")
    print(f"tbl_kalman_speed mean mde by sigma: {np.round(np.nanmean(mat_tks, axis=0), 4)}")
    print(f"grid: {full_grid}")
    print(f"\nfor reference, CV crossing was 1.0552, 95% CI (1.0209, 1.0881)")


if __name__ == "__main__":
    main()
