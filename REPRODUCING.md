# Reproducing and repository layout

Moved out of README.md to keep that file short; nothing here is new.
See README.md for the project itself.

---

## Reproducing

Budget **~2.5–3 hours end to end** on CPU-only hardware comparable to this
project's (Apple Silicon). ~1.6 hours of that is measured model-training
time (the five lines below marked "measured"); the remaining ~15 physics/
sweep scripts are unmeasured individually but are lighter-weight than
training — see the runtime note after the code block.

Run scripts live in `scripts/`; the library modules they import
(`config.py`, `kalman.py`, ...) stay at the repo root. `PYTHONPATH=.` on
every invocation below puts the repo root on the import path so those
imports resolve; run everything from the repo root, which is also what
keeps the `data/`/`outputs/` paths (relative to the current directory,
not to the script) working unchanged.

```bash
# environment — requires a Kaggle account.
# Python 3.14.7 is what this project was actually built and run on; no
# other version was tested, so none is claimed to work.
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # pinned to the versions actually run

# data preparation — no separate prep script, and no Kaggle API script is
# included here (this project downloaded manually, not via the Kaggle
# CLI). Sign in to Kaggle, open the NFL Big Data Bowl 2026 (prediction)
# competition at
#   https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction
# download its train/ folder, and place the files at data/train/
# (input_2023_wNN.csv + output_2023_wNN.csv, NN = 01..18).
# cache_pipeline.py builds and caches per-week parquet/npz on first use.

# sanity checks — fast, no Kaggle data needed for the two synthetic ones
PYTHONPATH=. python scripts/inspect_data.py     # confirm the schema on disk
PYTHONPATH=. python scripts/test_kalman.py      # synthetic sanity checks for the Kalman filter
PYTHONPATH=. python scripts/test_imm.py         # synthetic sanity checks for the IMM

# Part 1 — physics baselines and degradation sweep
PYTHONPATH=. python scripts/run_tier0a.py                      # Tier 0a/0b: baselines, Kalman, IMM, NIS check
PYTHONPATH=. python scripts/run_tier1_sweep.py                 # Task B: coarse degradation grid, all baselines
PYTHONPATH=. python scripts/run_tier1_crossover.py             # Task D/E/F: fine-grid crossover + bootstrap CI
PYTHONPATH=. python scripts/run_tier1_twoaxis.py               # Task G/H/I: second (ball-land) degradation axis
PYTHONPATH=. python scripts/run_tier1_r_convention_check.py    # floor vs quadrature R, both conventions reported
PYTHONPATH=. python scripts/run_tier1_taskn.py                 # Task N: coverage audit, picks the 1.9s primary horizon
PYTHONPATH=. python scripts/run_tier1_taskO.py                 # Task O: restates every Part 1 finding at 1.9s + 3s

# Part 2 — model training, cross-validated (CPU; each full run is ~8-28 min — see note)
PYTHONPATH=. python scripts/run_tier1_taskj.py                 # expand to 18 weeks, write outputs/splits.json
PYTHONPATH=. python scripts/run_tier1_taskp.py                 # write outputs/folds.json, out-of-fold baseline bar
PYTHONPATH=. python scripts/run_tier1_taskk.py                 # train K1/K2/K3, 5 folds each      (measured: ~28 min)
PYTHONPATH=. python scripts/run_tier1_taskk4.py                # train K4, 5 folds                 (measured: ~10 min*)
PYTHONPATH=. python scripts/run_tier1_k5.py                    # train K5, 5 folds                 (measured: ~8 min)
PYTHONPATH=. python scripts/run_tier1_taskl.py                 # full 13-baseline + K1-K3 CV table, per-role
                                                                 #                                   (measured: ~28 min)
PYTHONPATH=. python scripts/run_tier1_taskm.py                 # degradation sweep, K1-K3 + baselines, CV folds only
PYTHONPATH=. python scripts/run_tier1_taskk4_degrade.py        # degradation sweep, K4
PYTHONPATH=. python scripts/run_tier1_k5_degrade.py            # degradation sweep, K5

# Test-split evaluation — run ONCE. Everything above must be finished and
# the variant list frozen before this line runs; it is not a selection step.
PYTHONPATH=. python scripts/run_tier1_test_pass.py             # train final K1-K5 on all 231 train+val games,
                                                                 # score every variant + baseline on the untouched
                                                                 # 41-game test split, one pass    (measured: ~20 min)
PYTHONPATH=. python scripts/run_tier1_test_degrade.py          # the same degradation sweep, repeated on test
PYTHONPATH=. python scripts/plot_headline_crossing.py           # regenerates the Part 2 figure from
                                                                 # tier1_test_degrade_summary.csv, written above

# Part 3 — local half of the serving-latency measurement (CPU inference
# floor only; no AWS access needed for this one script)
PYTHONPATH=. python scripts/run_tier2b_phase1_local.py         # K3/K5/tbl_kalman_speed local inference timing
```

The rest of Part 3 (the round-trip, region, batch-size, throughput, and
cost numbers) was measured against a live SageMaker endpoint and is not
clone-and-run — there is no script that reproduces the latency table
itself, only the local floor above that it's compared against.

Runtimes marked "measured" are wall-clock times observed on this machine
(Apple Silicon, CPU only) during development; everything else scales with
the same per-fold/per-sigma-combo cost and wasn't separately re-timed, so
treat unmarked lines as the same order of magnitude, not as measured facts.
*One K4 fold's wall-clock printed as ~7,900s during a run on this machine —
almost certainly a system sleep/contention artifact, not real compute (its
epoch count and result were normal); the ~10 min figure excludes that
anomaly. The leakage audit, frozen-R check, gain measurements, and
effective-window test (a dozen-plus smaller scripts in `scripts/`, each
named `run_tier1_*`) aren't included above — they're forward-pass-only checks
against already-trained checkpoints, not part of the path to the headline
numbers. See Repository layout below for what each one answers.

Hardware: CPU only. No GPU is used anywhere — at 18k parameters and ~7,600
training samples, transfer overhead exceeds the benefit. Peak memory 565 MB
across 18 weeks of data, via column-selective reads and per-week caching.

---

## Repository layout

**Core library** (imported everywhere below, nothing here runs on its own)
- `config.py` — paths, weeks, split/fold seeds, noise grids, every tuned constant
- `data_loader.py` — CSV loading, play-direction mirroring, kinematics, `PlaySample`
- `cache_pipeline.py` — per-week parquet/npz caching for the full 18-week load
- `degrade.py` — sensor-noise injection (player-position and ball-land axes)
- `baselines.py` — six closed-form physics predictors
- `kalman.py` — hand-written Kalman filter (position-only and velocity-observing)
- `imm.py` — the 2-model IMM and the constant-acceleration Kalman variant
- `hybrid.py` — the 2×2 `toward_ball_land` / Kalman-posterior hybrid family
- `evaluate.py` — scoring and the coverage-gated summary tables
- `model.py` — feature engineering (incl. K4's Kalman-velocity variant), the GRU, masked loss

All run scripts below live in `scripts/`, invoked as `PYTHONPATH=. python
scripts/<name>.py` from the repo root — see Reproducing. Library modules
(above) stay at the repo root and are imported by name (`from config
import ...`), which is what `PYTHONPATH=.` is for.

**Part 1 — the physics study**
- `scripts/run_tier0a.py` — Tier 0a/0b baselines, Kalman, IMM, the NIS calibration check
- `scripts/run_tier1_sweep.py` — Task B: coarse degradation grid, every baseline
- `scripts/run_tier1_crossover.py` — Task D/E/F: fine-grid crossover sigma + bootstrap CI
- `scripts/run_tier1_twoaxis.py` — Task G/H/I: the ball-land degradation axis
- `scripts/run_tier1_r_convention_check.py` — floor vs. quadrature R, side by side (adopted quadrature)
- `scripts/run_tier1_taskn.py` — Task N: coverage audit; picks 1.9s as the honest primary horizon
- `scripts/run_tier1_taskO.py` — Task O: restates every earlier finding at 1.9s with coverage attached

**Part 2 — the learned models**
- `scripts/run_tier1_taskj.py` / `scripts/run_tier1_taskj_bonus.py` — expand to 18 weeks; re-run Part 1 on it
- `scripts/run_tier1_taskp.py` — writes `outputs/folds.json`; the out-of-fold baseline bar K1-K5 beat
- `scripts/run_tier1_taskk.py` — shared training/checkpointing/eval code; trains K1, K2, K3
- `scripts/run_tier1_taskk4.py` / `scripts/run_tier1_taskk4_degrade.py` / `scripts/run_tier1_taskk4_role.py` — K4
- `scripts/run_tier1_k5.py` / `scripts/run_tier1_k5_degrade.py` — K5 (on-the-fly noise augmentation)
- `scripts/run_tier1_taskl.py` — full 13-baseline + K1-K3 CV comparison table, per-role
- `scripts/run_tier1_taskm.py` — the degradation sweep that found the sigma=0.2 collapse

**The leakage audit and mechanism checks** (why the headline numbers are trusted)
- `scripts/run_tier1_taskk_checkB.py` / `_check1_2.py` / `_close.py` / `_permute.py` / `_permute2.py` —
  metric/alignment/input-window audit and the label-permutation negative controls
- `scripts/run_tier1_taskm_diag.py` — fine sigma grid, gate saturation, output-magnitude checks
- `scripts/run_tier1_taskm_gain.py` / `scripts/run_tier1_gain_full.py` — the velocity-gain measurement (K1/K3/K4/K5)
- `scripts/run_tier1_frozenr.py` / `_bystep.py` — is Kalman-family robustness structural or "being told"?
- `scripts/run_tier1_k3k4_crossover.py` / `scripts/run_tier1_k5_tks_crossover.py` — the two crossing-sigma bootstraps
- `scripts/run_tier1_k5_closing_check.py` / `_check2.py` — did K5 converge on `toward_ball_land`? (no)
- `scripts/run_tier1_effective_window.py` — the fixed-ramp smoothing-window test

**Test split**
- `scripts/run_tier1_test_pass.py` — the single frozen-model evaluation against the 41 held-out games
- `scripts/run_tier1_test_degrade.py` — the degradation sweep repeated on that same split
- `scripts/run_tier1_test_degrade_allbaselines.py` — the remaining 10 baselines, scored under noise
  on test for the first time, verifying the crossing against the full 13-baseline envelope

**Tests and utilities**
- `scripts/test_kalman.py`, `scripts/test_imm.py`, `scripts/inspect_data.py` — synthetic sanity
  checks and a raw-schema check; in `scripts/` (each has a `__main__` block, invoked directly)
- `scripts/smoke_test_degrade.py`, `scripts/imm_divergence_check.py` — unit/sanity checks; also in
  `scripts/` because both import from `scripts/run_tier1_sweep.py`
- `scripts/plot_headline_crossing.py` — regenerates the Part 2 figure (`tier1_test_headline_crossing.png`)
  from the committed `tier1_test_degrade_summary.csv`
- `animate.py` — library module (no `__main__`), imported by `scripts/run_tier0a.py` to produce
  `outputs/play_animation.gif`; stays at the repo root

**Outputs** (`outputs/`)
- `splits.json`, `folds.json`, `final_val_games.json` — every frozen game-ID partition, by seed
- `checkpoints/` — every trained model: 5 CV folds × 5 variants, plus the 5 final test-split models
- `*.csv` — every numeric result in this document, one file per script that produced it
- Not committed: checkpoints and the large per-play detail tables (>2 MB) —
  both are regenerated by the scripts above, not evidence meant to be read
  directly. Figures and small summary CSVs are committed.

**Serving artifact** (`deploy_k5/`)
- The one checkpoint that isn't a regenerable intermediate: the packaged
  K5 model actually deployed for Part 3 (`model.pth`, `model.tar.gz`,
  `code/inference.py`), 148 KB. Unlike `outputs/checkpoints/` — 32
  training-run checkpoints, reproducible by re-running Part 2 — this is
  the only way to reproduce Part 3's serving numbers without retraining,
  so it's committed.
