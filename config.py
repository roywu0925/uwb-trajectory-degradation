"""Tier 0a configuration.

Schema is now known from the real Kaggle files, so nothing guesses any more.

Data layout expected:
    data/train/input_2023_wNN.csv    tracking BEFORE the throw
    data/train/output_2023_wNN.csv   true positions WHILE the ball is in air
    data/supplementary_data.csv      (not used at Tier 0a)
"""

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
OUTPUT_DIR = Path("outputs")

# ---------------------------------------------------------------- field

FIELD_LENGTH = 120.0   # yards, including both end zones
FIELD_WIDTH = 53.3     # yards
FRAME_RATE = 10.0      # Hz
DT = 1.0 / FRAME_RATE

# ---------------------------------------------------------------- subset

# Weeks to load. Each input file is ~45 MB, so start with one or two and
# raise this once you know how long a run takes.
#   ["w01"]                 fast, good for iterating
#   ["w01", "w02", "w03"]   a reasonable Tier 0a sample
#   []                      every week found on disk
# All 18 weeks of the 2023 season as of Tier 1 Task J (TIER1_SPEC_4.md) --
# 128 plays (one week) was too few to train on. Use cache_pipeline's
# per-week caching, not data_loader.load_raw directly, when loading all
# of these at once; see that module's docstring for why.
WEEKS = [f"w{n:02d}" for n in range(1, 19)]

# Ball flight length, in frames, from num_frames_output.
# Long flights are the hard, interesting cases -- physics has more time to
# be wrong, and the receiver has more time to do something intentional.
MIN_HORIZON = 15    # >= 1.5 s of ball flight
MAX_HORIZON = 40    # sanity cap

MAX_PLAYS = None    # e.g. 500 to subsample; None keeps every play

# Restrict to certain player_role values, or None for all four.
# Roles seen in the data: 'Targeted Receiver', 'Defensive Coverage',
# 'Passer', 'Other Route Runner'.
ROLES_OF_INTEREST = None

# ---------------------------------------------------------------- plotting

# Stop drawing an error curve once fewer than this fraction of plays still
# have data at that horizon step -- otherwise the tail is a handful of
# outlier plays and reads as noise.
MIN_COVERAGE = 0.25

# ---------------------------------------------------------------- kalman (Tier 0b)

# UWB tag position-measurement noise variance. "6 inches (~0.17 yd)" was
# carried in this comment as "NGS/Zebra's public figure" -- it is uncited
# background, not a verified vendor figure; NFL Football Operations
# publishes no numeric accuracy spec (checked directly, 2026-10-31; see
# PART1_DETAIL.md's methodology note). This constant was chosen more
# conservative than that uncited figure, assuming real match conditions
# are worse. It is a config constant, not a fitted parameter -- see
# TIER0B_SPEC.md on why it must not be tuned against the eval score. The
# published position stream was later measured directly and found
# quantized to 0.01 yd with no detectable noise above that step -- this
# constant does not reflect that measurement and should not be read as
# "the sensor's real noise."
KALMAN_R_POS = 0.25 ** 2   # yd^2, sigma ~= 0.25 yd, an assumed design constant

# Measurement noise on the NGS-reported velocity (s, dir -> vx, vy), used
# only by the velocity-observing filter variant. No published spec for
# this figure; treated as a rough, deliberately un-tuned placeholder --
# Tier 1's R-sweep is where this gets examined properly, not here.
KALMAN_R_VEL = 0.5 ** 2    # (yd/s)^2, sigma ~= 0.5 yd/s

# Minimum history frames before the filter is trusted to have settled.
# Below this, fall back to a single-frame constant-velocity init instead
# of running predict/update (see kalman.py).
KALMAN_MIN_HISTORY = 3

# ---------------------------------------------------------------- imm (Tier 0b)

# Model transition matrix, PI[i, j] = P(model j at step k | model i at
# step k-1). Row/col 0 = "steady", row/col 1 = "manoeuvring". Sticky
# (diagonal-heavy) by design choice -- a receiver is expected to stay in
# whichever regime it's in most steps -- not fitted to this dataset and
# not tuned against results. See imm.py.
IMM_TRANSITION = np.array([[0.95, 0.05],
                           [0.10, 0.90]])

# The manoeuvring model's process-noise sigma_a, as a multiple of the
# steady model's (the NIS-calibrated sigma_a computed once per dataset in
# run_tier0a.py). "Roughly an order of magnitude" per TIER0B_SPEC.md;
# not adjusted against results.
IMM_MANOEUVRE_RATIO = 10.0

# Experiment B (CONTEXT.md): a re-specification of the steady model's
# sigma_a, run once, alongside (not replacing) the original.
#
# The original IMM run reused sigma_a = 1.791, the value NIS-calibrated
# for the *single*-model Kalman filter. That calibration was for one
# model covering all pre-throw motion -- constant-velocity stretches and
# real cuts both -- so it is optimal precisely because it already
# absorbs manoeuvres into "normal". Handing that same value to a model
# whose entire job is to represent *non*-manoeuvring motion is a
# specification error: it means the steady model is already permissive
# enough to explain real cuts, which is the direct cause of the 0.0%
# manoeuvring-model share observed with it.
#
# 0.3 is used here because it is what produced clean model separation in
# imm's synthetic test (test_imm.py) -- not picked by trying values
# against the real error table. This constant is set once, used once,
# and its result is reported whether it helps or not.
IMM_STEADY_SIGMA_A_EXPERIMENT = 0.3

# ---------------------------------------------------------------- degradation sweep (Tier 1 Task B)

# Position-noise standard deviations to sweep, in yards. 0.0 is the
# "clean" endpoint -- but see degrade.py: even at sigma=0.0, positions
# passed through finite differencing ("derived" velocity_source) do not
# reproduce Tier 0's numbers, because differencing amplifies whatever
# residual noise NGS's own smoothing left behind. Not tuned for a nice
# curve; see TIER1_SPEC.md.
NOISE_SIGMAS = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]   # yards

N_SEEDS = 3
DEGRADE_BASE_SEED = 20260923

# "derived": corrupt positions and re-derive vx/vy/speed/direction/ax/ay
# from them by finite differencing -- the realistic case, what a
# genuinely degraded sensor's own downstream processing would produce.
# "ngs": corrupt positions but leave speed/direction untouched at the
# clean NGS values, isolating "position noise hurts" from "velocity
# noise hurts". See degrade.py.
VELOCITY_SOURCES = ["derived", "ngs"]

# ---------------------------------------------------------------- filter R under degradation (Tier 1 Task C, TIER1_SPEC_2.md)

# Velocity measurement noise (sigma, yd/s) used by the filters' R matrix
# in "ngs" mode -- constant, independent of the position-noise sigma
# being swept, because the NGS s/dir columns are never corrupted in that
# mode (see degrade.py). NGS does not publish a velocity accuracy figure
# the way it does for position (KALMAN_R_POS above); this is a stated
# assumption, not a spec, and not tuned against results. "derived" mode
# does not use this constant -- its sigma_vel is derived analytically
# from sigma_pos each run (see run_tier1_sweep.set_filter_R).
DEGRADE_SIGMA_VEL_NGS = 0.3   # yd/s

# Fine grid around the constant_velocity/kalman crossover (Task D),
# which Task B's coarse grid [0, 0.2, 0.5, 1, 2, 5] located only as
# "somewhere below 0.2" -- below NGS's own published accuracy range.
CROSSOVER_SIGMAS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30]
CROSSOVER_SEEDS = 5

# ---------------------------------------------------------------- second degradation axis (Tier 1 Task G, TIER1_SPEC_3.md)

# Landing-point noise, yards -- ball_land_x/y corrupted the same way as
# player positions but along its own axis, to find out whether the
# hybrid predictor's near-immunity to sigma_player (Part 2) survives a
# landing point that is itself only estimated, not exact.
BALL_SIGMAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]   # yards

# Named separately from NOISE_SIGMAS per TIER1_SPEC_3.md, even though the
# values are identical -- this is the "existing full grid" on the other
# axis of the two-axis sweep, not a new choice.
PLAYER_SIGMAS = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]   # yards

GRID_SEEDS = 3

# Independent from DEGRADE_BASE_SEED on purpose: the ball-land noise and
# the player-position noise must be drawn from separate RNG streams, so
# that sigma_ball=0 is an exact (not approximate) no-op and reproduces
# every Part 2 number bit-for-bit -- see degrade.add_ball_land_noise.
BALL_NOISE_BASE_SEED = 20260927

# ---------------------------------------------------------------- train/val/test splits (Tier 1 Task J, TIER1_SPEC_4.md)

# By GAME, not by play or randomly -- plays from the same game share
# players, personnel and schemes, so a random play-level split leaks
# that structure into the test set. Roughly 70/15/15; the exact game IDs
# are fixed by this seed and written to outputs/splits.json so every
# later run (including Task K/L's model training) uses the identical
# partition. The test split is not to be touched until Task L.
SPLIT_SEED = 20260927
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
