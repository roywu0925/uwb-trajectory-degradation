"""Tier 1 Task P: the fold structure, built once and frozen
(TIER1_SPEC_5.md, amended by TIER1_SPEC_6.md).

    PYTHONPATH=. python scripts/run_tier1_taskp.py  # run from the repo root

Writes outputs/folds.json BEFORE any model training exists -- 5 grouped
folds (by game) over the 231 train+val games from outputs/splits.json,
fixed by seed. Every Task K model variant reads this same file; three
models trained on three different partitions would not be comparable,
and that failure would be silent.

Also computes and reports the out-of-fold baseline table -- every
baseline (the six physics predictors plus the Kalman/IMM/hybrid family)
scored on the full train+val play set, under the `quadrature` R
convention (which, at sigma=0 -- there is no degradation in Task P/K/L,
only Task M -- is exactly config.KALMAN_R_POS/KALMAN_R_VEL, i.e. the
module defaults kalman.py and imm.py already start with; no
set_filter_R call is needed here at all). Baselines have no training
step, so "out-of-fold" for them just means "the exact same play set the
models' out-of-fold predictions will cover" -- this table is the bar
every K1/K2/K3 result gets compared against.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from baselines import BASELINES
from cache_pipeline import build_dataset_cached
from config import DT, OUTPUT_DIR, SPLIT_SEED, WEEKS
from evaluate import evaluate, summarise_at_step
from hybrid import make_hybrid_family
from imm import make_imm, make_kalman_ca
from kalman import build_Q, calibrate_sigma_a, estimate_sigma_a, make_kalman

N_FOLDS = 5
PRIMARY_STEP = 19   # 1.9s, run_tier1_taskn.py's coverage-chosen primary horizon
STEP_3S = 30


def build_folds(train_val_games: list[int], seed: int, n_folds: int) -> list[list[int]]:
    games = np.array(sorted(train_val_games))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(games)
    folds = [shuffled[i::n_folds].tolist() for i in range(n_folds)]
    return [sorted(int(g) for g in fold) for fold in folds]


def assert_folds_valid(folds: list[list[int]], train_val_games: list[int]) -> None:
    seen = set()
    for fold in folds:
        fold_set = set(fold)
        assert not (seen & fold_set), (
            f"game(s) {seen & fold_set} appear in more than one fold")
        seen |= fold_set
    assert seen == set(train_val_games), (
        f"fold union does not match train+val games exactly -- "
        f"missing {set(train_val_games) - seen}, extra {seen - set(train_val_games)}")


def fold_table(samples, folds: list[list[int]]) -> pd.DataFrame:
    rows = []
    for i, game_ids in enumerate(folds):
        game_set = set(game_ids)
        sub = [s for s in samples if s.game_id in game_set]
        n_tracks = sum(len(s.state) for s in sub)
        n_primary = sum(len(s.state) for s in sub if s.horizon >= PRIMARY_STEP)
        n_3s = sum(len(s.state) for s in sub if s.horizon >= STEP_3S)
        rows.append({"fold": i, "games": len(game_ids), "plays": len(sub),
                    "tracks": n_tracks, "tracks_at_primary": n_primary,
                    "tracks_at_3s": n_3s})
    return pd.DataFrame(rows)


def build_predictors_quadrature(sigma_a: float) -> dict:
    """quadrature at sigma=0 IS config.KALMAN_R_POS/KALMAN_R_VEL exactly
    -- the module defaults kalman.py/imm.py already start with, before
    any set_filter_R call. No degradation happens in Task P/K/L (only
    Task M sweeps sigma_player), so this is just "build the predictors
    with R untouched" -- stated explicitly so a reader doesn't wonder
    why set_filter_R never appears here.
    """
    Q = build_Q(sigma_a)
    predictors = dict(BASELINES)
    predictors["kalman"] = make_kalman(Q, use_velocity=False)
    predictors["kalman_velocity"] = make_kalman(Q, use_velocity=True)
    predictors["kalman_ca"] = make_kalman_ca(sigma_a)
    predictors["imm"] = make_imm(sigma_a)
    predictors.update(make_hybrid_family(Q))
    return predictors


def main():
    print(f"loading {len(WEEKS)} weeks via cache_pipeline...")
    samples, _ = build_dataset_cached(WEEKS)
    print(f"{len(samples):,} plays total\n")

    with open(OUTPUT_DIR / "splits.json") as f:
        splits = json.load(f)
    train_val_games = splits["train"] + splits["val"]
    print(f"train+val games: {len(train_val_games)}")

    folds = build_folds(train_val_games, SPLIT_SEED, N_FOLDS)
    assert_folds_valid(folds, train_val_games)
    print(f"{N_FOLDS} folds built, no overlap, union matches train+val exactly")

    with open(OUTPUT_DIR / "folds.json", "w") as f:
        json.dump({"seed": SPLIT_SEED, "n_folds": N_FOLDS, "folds": folds}, f, indent=2)
    print(f"wrote {OUTPUT_DIR / 'folds.json'}")

    print("\nfold table:")
    ftable = fold_table(samples, folds)
    print(ftable.to_string(index=False))

    train_val_set = set(train_val_games)
    train_val_samples = [s for s in samples if s.game_id in train_val_set]
    print(f"\n{len(train_val_samples):,} train+val plays "
         f"({sum(len(s.state) for s in train_val_samples):,} tracks) "
         f"-- the out-of-fold evaluation set for every baseline and every "
         f"Task K model variant")

    sigma_a_raw = estimate_sigma_a(train_val_samples)
    sigma_a = calibrate_sigma_a(train_val_samples, sigma_a_hi=sigma_a_raw, target=2.0)
    print(f"\nsigma_a = {sigma_a:.4f} (raw {sigma_a_raw:.4f}), "
         f"estimated on train+val only -- not the test-inclusive full-data "
         f"value used in Tier 1 Part 1, since test stays untouched")

    predictors = build_predictors_quadrature(sigma_a)
    errors = evaluate(train_val_samples, predictors)

    print("\n" + "=" * 70)
    print("Out-of-fold baseline bar -- train+val plays, quadrature convention")
    print("=" * 70)
    for step, label in ((PRIMARY_STEP, "primary (1.9s)"), (STEP_3S, "secondary (3.0s)")):
        print(f"\n--- {label} ---")
        t = summarise_at_step(errors, step)
        print(t.to_string(index=False))
        t.to_csv(OUTPUT_DIR / f"tier1_taskp_baseline_bar_step{step}.csv", index=False)


if __name__ == "__main__":
    main()
