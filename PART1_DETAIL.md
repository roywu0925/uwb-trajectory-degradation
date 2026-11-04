# Trajectory prediction under degraded sensing

Short-horizon multi-target trajectory prediction on 10 Hz UWB player
tracking data (NFL Next Gen Stats), comparing model-based estimation
against learned sequence models and characterising how each degrades as
the sensing quality drops.

**Status: Tier 1 Part 1 — sensor degradation sweep complete. Run on the
real 2023 season data; see [Results](#results) below.**

---

## The question

At the moment the ball leaves the quarterback's hand, given every player's
position, speed and heading up to that instant, where will the tracked
players be while the ball is in the air?

The interesting quantity is not the accuracy number. It is the split:

- **How much of player motion is explained by physics alone?**
  A body with a position and a velocity keeps going. That costs nothing to
  compute and it accounts for some of the motion.
- **How much is explained by intent?**
  A receiver running a route, a defender reacting to the throw — that is
  not physics, and no filter recovers it. It has to be learned.

Tier 0 measures the first so that later tiers can measure the second.
Without a physics floor, a learned model's error is a number with nothing
to compare it against.

## Roadmap

| Tier | Content | Output |
|---|---|---|
| 0a | Data pipeline, physics baselines, animation | error-vs-horizon curve |
| 0b | Kalman / IMM filter, state estimation | filter vs naive extrapolation |
| 1 | Learned sequence model + sensing degradation study | robustness curves |
| 2 | Streaming inference, quantisation, latency benchmark | latency vs accuracy |

## Why UWB matters here

Next Gen Stats positions come from ultra-wideband tags in the players'
shoulder pads, read by receivers around the stadium at roughly 10 Hz. They
are measurements, not ground truth: every sample carries localisation
error, and coverage degrades when bodies occlude a tag.

Three consequences shape the project:

1. **Velocity estimated by differencing positions amplifies that error**,
   because the sampling interval is small. Standard motivation for a state
   estimator, and the reason Tier 0b exists.
2. **Sensing quality is a variable, not a constant.** Tier 1 treats it as
   one: inject localisation noise, drop samples, reduce the update rate,
   and measure the accuracy cost. That trade-off curve is the main result.
3. **The ball's landing point is itself an estimate.** It is supplied as an
   exact input here, but a live system would infer it from the same noisy
   tracking stream — so error in the landing point is a second degradation
   axis, and one the baselines below are directly sensitive to.

## Data

The competition files do the hardest part of the setup:

```
data/train/input_2023_wNN.csv     tracking up to the throw
data/train/output_2023_wNN.csv    true positions while the ball is in air
data/supplementary_data.csv       not used by this pipeline
```

The prediction anchor needs no guessing — the input file ends at the throw
and the output file begins there. Useful columns:

| Column | Role |
|---|---|
| `player_to_predict` | only these players have ground truth |
| `num_frames_output` | per-play prediction horizon, 6–31 frames |
| `player_role` | targeted receiver / coverage / passer / other route runner |
| `ball_land_x`, `ball_land_y` | where the ball comes down, known up front |
| `s`, `a`, `dir`, `o` | speed, acceleration magnitude, heading, orientation |

Horizons vary per play, so every result is reported per horizon step with
the number of plays still contributing at that step.

## Running it

```bash
pip install -r requirements.txt

# run scripts live in scripts/; PYTHONPATH=. so they can still import the
# library modules (config.py, kalman.py, ...), which stay at the repo root
PYTHONPATH=. python scripts/inspect_data.py     # confirm the schema on disk
PYTHONPATH=. python scripts/test_kalman.py      # synthetic sanity checks for the Kalman filter
PYTHONPATH=. python scripts/test_imm.py         # synthetic sanity checks for the IMM
PYTHONPATH=. python scripts/run_tier0a.py       # Tier 0a + 0b: physics baselines and filters, clean data
PYTHONPATH=. python scripts/run_tier1_taskn.py       # Tier 1: coverage audit, picks the primary horizon
PYTHONPATH=. python scripts/smoke_test_degrade.py    # sanity checks before the degradation sweep
PYTHONPATH=. python scripts/run_tier1_sweep.py       # Tier 1: coarse degradation sweep (0-5 yd)
PYTHONPATH=. python scripts/run_tier1_crossover.py   # Tier 1: fine sweep around the crossover (0-0.3 yd)
PYTHONPATH=. python scripts/imm_divergence_check.py  # IMM manoeuvring-probability vs. sensor noise
PYTHONPATH=. python scripts/run_tier1_twoaxis.py     # Tier 1: player-noise x landing-point-noise grid, dual horizon
```

Despite the name, `scripts/run_tier0a.py` runs both Tier 0a's physics baselines
and Tier 0b's Kalman/IMM filters against clean data in one pass — the
name is a holdover from before Tier 0b existed.

`WEEKS` in `config.py` now covers all 18 weeks of the 2023 season by
default (each week is about 45 MB of input CSV, ~800 MB combined).
`cache_pipeline.py` loads one week at a time and caches the built
`PlaySample`s to `data/cache/` (parquet + `.npz`) so this doesn't need
to happen more than once — peak memory is ~565 MB, not the multi-GB a
naive all-at-once `read_csv` would hit, mostly from `data_loader`'s
`usecols=` trim of columns nothing in the pipeline reads. To iterate
faster during development, set `WEEKS = ["w01"]` and expect Tier 0/1
numbers to differ from what's reported here, which was run on the full
set. Outputs land in `outputs/`:

- `error_curves.png` — mean displacement error against time since the throw
- `error_by_role.csv` — which roles each baseline handles badly
- `baseline_errors.csv` — regenerated by `scripts/run_tier0a.py` (not committed,
  >2 MB — see Reproducing); the per-step, per-role numbers behind both
- `play_animation.gif` — one play, actual positions against predicted
- `horizon_coverage.csv`/`.png` — plays/tracks/coverage at every horizon
  step; this is what picked the 1.9s primary horizon (see
  [Tier 1: degradation sweep](#tier-1-degradation-sweep))
- `splits.json` — the fixed 70/15/15 train/val/test split, by game
- `imm_model_probabilities.csv`, `imm_model_probabilities_resigma.csv` —
  per-frame, per-player-track `mu_steady`/`mu_manoeuvre` for the IMM and
  its re-specified variant (see [Results](#results))
- `tier1_degradation.csv` (regenerated by `scripts/run_tier1_sweep.py`, not
  committed — see Reproducing), `tier1_robustness.png`/
  `tier1_robustness_full.png` — the coarse 0-5 yd degradation sweep
- `tier1_crossover.csv` (regenerated by `scripts/run_tier1_crossover.py`, not
  committed — see Reproducing), `tier1_crossover.png` — the fine 0-0.3 yd
  sweep around the constant_velocity/kalman crossing
- `imm_mu_vs_sigma.csv`, `imm_model_prob_vs_sigma.png` — mean
  P(manoeuvring) against sensor noise
- `tier1_two_axis.csv` (regenerated by `scripts/run_tier1_twoaxis.py`, not
  committed — see Reproducing), `tier1_two_axis.png`,
  `tier1_two_axis_heatmap.png`, `tier1_two_axis_imm_mu.csv` — the
  player-noise x landing-point-noise grid (Finding 4), dual-horizon

## Baselines

| Predictor | Assumption |
|---|---|
| `constant_position` | nobody moves — the absolute floor |
| `constant_velocity` | current velocity holds |
| `constant_acceleration` | current acceleration holds |
| `damped_velocity` | velocity decays with a fixed time constant |
| `toward_ball_land` | run straight at the landing spot at current speed |
| `velocity_turning_to_ball` | momentum early, destination late |

Two comparisons carry most of the information:

`constant_acceleration` against `constant_velocity` locates where a purely
kinematic description stops holding. Nobody sustains their instantaneous
acceleration for three seconds, so the quadratic term diverges; the
crossover point is a result, not a bug.

`toward_ball_land` against `constant_velocity` separates *where a player is
going* from *where they are pointed*. The first uses only the destination
and ignores heading; the second the reverse. Where the destination-only
baseline wins, intent is already legible in the geometry — and a learned
model has to beat that, not just beat constant velocity.

## Results

Week 1, 2023 season (`WEEKS = ["w01"]`): 128 plays, 526 player-tracks,
horizon 15–34 frames (median 1.9 s of ball flight).

Mean displacement error, yards (lower is better):

| predictor | overall | @1s | @2s | @3s |
|---|---|---|---|---|
| `constant_velocity` | 2.256 | 1.693 | 4.776 | 7.284 |
| `velocity_turning_to_ball` | 2.339 | 2.070 | 4.608 | **6.397** |
| `toward_ball_land` | 2.876 | 2.769 | 4.883 | 6.498 |
| `constant_acceleration` | 2.952 | 1.980 | 6.439 | 11.101 |
| `damped_velocity` | 3.678 | 2.588 | 8.249 | 14.670 |
| `constant_position` | 6.908 | 5.896 | 13.277 | 20.243 |

`constant_velocity` wins overall and through about 2s, but `toward_ball_land`
*alone* — no heading, no blending, just "walk straight at the known landing
spot" — already beats it by 3s (6.498 vs 7.284 yd). That is the real
finding: geometric destination information, on its own, becomes the
dominant signal at the far end of the horizon; it costs ~1s of extra error
early only because it ignores current heading entirely.
`velocity_turning_to_ball` blends in `constant_velocity` to cover exactly
that early gap and edges `toward_ball_land` out a little further (6.397 at
3s) — but the blend isn't where the win comes from, the destination
information is. `toward_ball_land` is also visibly bounded (it can't
overshoot the landing spot), unlike `constant_acceleration` and
`damped_velocity`, which diverge past ~1.5s.

**This sets the bar for Tier 1.** A model that only beats
`constant_velocity`'s 7.28 yd @3s hasn't learned anything a geometry-only
baseline didn't already know — `velocity_turning_to_ball` gets there with
zero parameters fit to data. The real target at 3s is **6.40 yd**, and
most of that number comes from the ball's landing point, not from any
kinematic modelling.

*(Both numbers above are week-1-only, at the 3s horizon — which turned
out to have only 4.6% track coverage on that sample. Tier 1's coverage
audit found this "destination is worth ~0.9 yd" figure is itself
horizon-dependent: ~0.3 yd at the honest 1.9s primary horizon, growing
past 1.5 yd by 3s, on the full 18-week dataset. See
[Tier 1: degradation sweep](#tier-1-degradation-sweep), Finding 2 — the
number here isn't wrong, but it's a far-horizon, thin-sample figure, not
the general answer.)*

By role, mean error (yards):

| predictor | Defensive Coverage | Targeted Receiver |
|---|---|---|
| `velocity_turning_to_ball` | 2.707 | 1.174 |
| `constant_velocity` | 2.340 | 1.991 |
| `toward_ball_land` | 3.310 | 1.499 |
| `constant_acceleration` | 2.921 | 3.048 |
| `damped_velocity` | 3.565 | 4.033 |
| `constant_position` | 6.391 | 8.545 |

*(These figures are week 1 only, matching the main results table above.
Recomputed on the full 18 weeks, every value moves by less than 0.3 yd and
the ordering is unchanged.)*

Every baseline that uses the landing point does *better* on the targeted
receiver than on the defender covering them — not worse. Giving every
player the same known destination turns out to be far more informative
for the player actually running to it than for the one reacting to the
throw in real time.

### Tier 0b: does state estimation beat naive extrapolation?

`constant_velocity` reads one frame. Every play carries 15–34 frames of
pre-throw history it never touches. Tier 0b (`kalman.py`, `imm.py`) asks
whether reading that history with a Kalman filter or an IMM does better —
at the 3 s horizon, where the comparison above already puts the real bar:

| predictor | uses ball landing point | velocity source | 3 s MDE (yd) |
|---|---|---|---|
| `velocity_turning_to_ball` | yes | NGS `s`/`dir` | 6.397 |
| `toward_ball_land` | yes | NGS `s` (magnitude only) | 6.498 |
| `constant_velocity` | no | NGS `s`/`dir` | 7.284 |
| `imm` / `kalman_ca` | no | estimated from positions (+ accel state) | 7.383 / 7.385 |
| `kalman_velocity` | no | positions + NGS `s`/`dir` as measurements | 7.715 |
| `kalman` | no | positions only | 8.616 |

**The split is information, not algorithm.** `constant_velocity` and
`velocity_turning_to_ball` take the same velocity input; the 0.887 yd
between them is entirely the landing point. Meanwhile every filter below
`constant_velocity` in the table is competing on "estimate current
velocity well," and none of them beats the naive method that just reads
NGS's own velocity columns.

**Why the filters lose.** NGS `s`/`dir` are not raw measurements — they
are the output of the vendor's own smoothing over the position sequence.
Re-estimating velocity from ~30 downstream positions reproduces work
already done with better inputs. Evidence: adding `s`/`dir` as
measurements (`kalman_velocity`) improves the position-only filter from
8.616 to 7.715 yd.

**IMM gained nothing from switching.** IMM reached 7.383 yd, but a
standalone 6-dimensional constant-acceleration Kalman filter
(`kalman_ca`) reproduced it to within 0.002 yd, and the manoeuvring model
won 0.0% of frames. The improvement is model order, not regime switching.

**The gap narrows with horizon, and connects to the NIS finding.** None
of the four filters ever wins on clean data at any horizon — but how
badly they lose shrinks sharply as the horizon extends:

| predictor | overall | @1s | @2s | @3s |
|---|---|---|---|---|
| `constant_velocity` | 2.256 | 1.693 | 4.776 | 7.284 |
| `imm` | 2.333 | 1.770 | 4.887 | 7.383 |
| `kalman_ca` | 2.334 | 1.771 | 4.889 | 7.385 |
| `kalman_velocity` | 2.883 | 2.342 | 5.738 | 7.715 |
| `kalman` | 3.720 | 3.182 | 6.959 | 8.616 |

`kalman` is 88% worse than `constant_velocity` at 1s and only 18% worse
at 3s; `kalman_velocity` goes from 38% worse to 6% worse; `imm`/`kalman_ca`
from ~5% worse to ~1% worse. This matches the NIS finding directly: the
filter's estimate is a smoothed, slightly-lagged read of velocity, and
that lag costs the most right where a single fresh frame
(`constant_velocity`'s entire strategy) has had the least time to be
wrong. As the horizon extends, `constant_velocity`'s own error compounds
too, and the filter's smoothing stops being a net liability as fast.

The honesty constraints that make this comparison defensible:

- `R` came from an assumed design constant (σ ≈ 0.25 yd, `KALMAN_R_POS`),
  not from tuning and not from a published spec — NFL Football Operations
  publishes no numeric accuracy figure for NGS (see the Methodology note
  below on what was later measured directly from the data).
- `Q`'s starting point (`sigma_a = 2.388 yd/s²`) was estimated once from
  one-step constant-velocity residuals on the input frames, printed, and
  never touched against `output_*.csv`.
- NIS — an answer-blind diagnostic, using only the filter's own priors
  and measurements — said that starting `Q` was inflated (NIS ≈ 1.27
  against an expected 2). The NIS-calibrated `Q` (`sigma_a = 1.791`) made
  the 3 s error *worse*, not better (`kalman`: 8.369 → 8.616 yd). Kept
  anyway. It's the sharpest single finding in Tier 0b: one-step filter
  consistency and three-second extrapolation want opposite memory
  lengths, so a diagnostic that is correct on its own terms can still
  point the wrong way for the actual objective.

**This result is conditional on today's data quality, not a verdict on
state estimation.** Every number above holds on the published NGS stream as
it actually arrives — already quantized to 0.01 yd with no detectable noise
above that (measured directly; see the Methodology note further down), not
assumed from a vendor spec. Tier 1 (below) tests what happens as noise is
added on top of that already-processed stream, and the answer reverses
fast: filtering stops being pointless well before much noise has been
added.

### Tier 1: degradation sweep

Tier 0b's result holds at exactly one sensor quality: NGS's own. This
tier corrupts the input positions with controlled Gaussian noise
(`degrade.py`, `scripts/run_tier1_sweep.py`, `scripts/run_tier1_crossover.py`,
`scripts/run_tier1_twoaxis.py`) and re-runs every predictor unchanged, to find
out whether — and where — that result survives. Every number below is
from the full 18-week dataset (2,729 plays, 11,254 player-tracks, 272
games) unless marked otherwise, and every number carries its coverage —
see the methodology note below on why that qualifier is not optional.

**Which `R` convention, and why it matters before any headline number can
be stated.** Every Kalman-family predictor's measurement noise `R` is
built from a position-noise scale `σ_pos`. Two ways to build it from the
injected sweep noise `σ`: `floor` (`σ_pos = max(σ, 0.05)`, used through
the rest of this section's detailed sigma-by-sigma tables) and
`quadrature` (`σ_pos = sqrt(KALMAN_R_POS + σ²)`). **This document adopts
`quadrature` as primary**, for three reasons, in this order:

1. **It's the more conservative one.** In `derived` mode, `σ` is injected
   on top of positions from the *published* stream, which this project's
   own later measurement (see Methodology) found to be quantized to 0.01 yd
   with no detectable noise above that — not the `σ≈0.25` this project had
   assumed as a design constant (`KALMAN_R_POS`) before that measurement
   existed. Telling the filter `R = 0.05²` at `σ=0` understates even the
   assumed figure; `quadrature` was the more conservative choice given what
   was known at the time it was adopted, not a claim that 0.25 was ever a
   confirmed real noise figure.
2. **Continuity.** `quadrature` makes the sweep's `σ=0` point reproduce
   the un-degraded Tier 0b filter *exactly* — `KALMAN_R_POS + 0² =
   KALMAN_R_POS`. It reproduces the standalone Tier 0b `tbl_kalman_speed`
   clean number, 5.245 yd, to three decimals. A degradation sweep whose
   zero point doesn't reproduce the experiment it's extending is a
   different experiment; that exact match is what continuity means, not
   a coincidence.
3. **It's the less flattering choice, which is itself evidence.**
   `quadrature` makes every Kalman-family clean-data number *worse*
   (`tbl_kalman_speed`: 4.77 → 5.25 yd) and every crossing *later*
   (`kalman`: 0.105 → 0.147; `tbl_kalman_speed`: 0.028 → 0.075) — it was
   not selected because it produced a better-looking result.

`floor` is kept throughout the rest of this section as a stated
sensitivity analysis, not deleted — every detailed σ-by-σ table below
still uses it, and is labelled as such.

**The headline: clean-data rankings are not just incomplete, they are
actively misleading.** At the honest primary horizon (1.9s, 53%
coverage), under the coupled degradation scenario (`σ_ball = σ_player`,
i.e. landing-point error scaling with player-position error), `σ=0`
column under `quadrature`:

| predictor | σ=0 (clean) | σ=5 (coupled) |
|---|---|---|
| `velocity_turning_to_ball` | **4.26** (1st) | 20.00 (**last**) |
| `toward_ball_land` | 4.54 | 13.04 |
| `tbl_kalman_pos` | 4.71 | 13.06 |
| `tbl_kalman_speed` | 5.25 (4th) | **9.19** (**1st**) |
| `tbl_kalman_both` | 5.45 | 11.43 |

*(The `σ=5` column is from the two-axis grid, run under `floor` and not
re-run under `quadrature` — but the two conventions converge at high σ:
`sqrt(0.25² + 5²) = 5.006` vs. `max(5, 0.05) = 5.000`, a 0.1% difference,
so this column is convention-independent for practical purposes. Only
the low-σ region, where this section's convention choice actually
matters, needed re-running.)*

The clean-data winner becomes the worst method under realistic sensing,
and the clean-data 4th-place method becomes the best — and under the
physically-correct convention, the gap by which `tbl_kalman_speed`
starts behind is *larger* (0.68 yd, not 0.21) than first reported, which
only sharpens the inversion. Mechanism: `velocity_turning_to_ball`
blends in `constant_velocity`'s heading, and heading (derived from
position differencing) is the single most noise-fragile quantity in
this entire study — it's what wins on clean data and what collapses
fastest once the sensor degrades.

**Finding 1 — the crossover is a curve, not a constant, and the
headline number is the hybrid's, not the raw filter's.** `tbl_kalman_speed`
is the method this project actually recommends, so its crossing against
`constant_velocity` is the one to lead with, not plain `kalman`'s:

| | `tbl_kalman_speed` crossing | `kalman` crossing |
|---|---|---|
| `quadrature` (primary) | **0.075 yd** | 0.147 yd |
| `floor` (sensitivity) | 0.028 yd | 0.105 yd |

`tbl_kalman_speed` crosses at 0.075 yd under `quadrature`, 0.028 yd under
`floor`; plain `kalman` crosses later, at 0.147 and 0.105 respectively.
These stand as measurements. What they mean relative to the real sensor's
noise is not stated here — this project later measured the published
position stream directly (see Methodology) and found no basis for comparing
a crossing sigma to "NGS's accuracy": there is no confirmed published
figure, and the published stream itself carries no detectable noise above
its 0.01 yd quantization step, which is not the same quantity as whatever
noise the raw sensor has. Earlier drafts of this document compared these
crossings to an assumed "NGS published range of 0.15–0.3 yd"; that figure
was never sourced and has been removed rather than replaced.

**A second crossing, same hybrid, different opponent.** The 0.075 figure
above is `tbl_kalman_speed` against `constant_velocity`. A different,
cleaner comparison isolates the cost of smoothing specifically:
`tbl_kalman_speed` vs. `toward_ball_land` are identical in direction
source (both point at the ball's landing spot) and differ only in where
speed comes from — a Kalman posterior vs. NGS's own reported speed. On
clean, held-out test data that costs **0.77 yd**. Run on the same fine
grid (`CROSSOVER_SIGMAS`, extended to σ=1.0), `quadrature` convention,
3 seeds per nonzero σ, train+val population:

| σ | 0.0 | 0.02 | 0.05 | 0.08 | 0.1 | 0.13 | 0.16 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `toward_ball_land` | 4.44 | 4.46 | 4.57 | 4.75 | 4.91 | 5.16 | 5.44 | 5.82 | 6.27 | 6.66 | 7.36 | 7.95 | 8.44 | 9.15 | 9.59 |
| `tbl_kalman_speed` | 5.20 | 5.20 | 5.21 | 5.23 | 5.24 | 5.27 | 5.30 | 5.36 | 5.43 | 5.51 | 5.67 | 5.83 | 5.98 | 6.26 | 6.49 |

Crossing: **σ = 0.1432**, 95% CI (0.1355, 0.1513) (1000-resample
play-level bootstrap, same method as the rest of this section) — about
**1.9× the `constant_velocity` crossing**, not the same number.

**What this crossing means relative to the real sensor's accuracy is
unknown**, and no claim is made here that it does or doesn't contain "the
sensor's rated noise" — that comparison depended on the assumed 0.15–0.30 yd
range, which this project later found had no source (see Methodology) and
removed rather than replaced. The crossing itself, 0.1432 (95% CI
0.1355–0.1513), and the 0.77 yd clean-data cost it's paired with, stand as
measurements. README.md's Finding 1 leads with both crossings (this one and
the `constant_velocity` one), each named with its pair, with no accuracy
comparison attached to either.

For historical comparison, the full-precision (5-seed, 1000-resample
bootstrap) numbers already reported for `kalman` under `floor` still
stand as measured:

| | point | 95% CI | width | coverage |
|---|---|---|---|---|
| Primary (1.9s) | **0.103 yd** | [0.100, 0.107] | 0.0066 | 53.0% (5,960 tracks) |
| Secondary (3.0s) | 0.057 yd | [0.046, 0.072] | 0.0257 | 3.5% (394 tracks) |
| *old (week 1, 3s)* | *0.029* | *[0.004, 0.068]* | *0.064* | *4.6% (24 tracks)* |

The crossing estimate itself moved from the old week-1 value (0.029) to
0.103 under `floor` to 0.147 under `quadrature` as the dataset and
convention were corrected — each correction moved the same direction
(later, not earlier), which is worth keeping even without a margin to
express it against. (An earlier version of this document expressed that
movement as a shrinking margin to "the NGS floor"; that comparison relied
on the unsourced 0.15–0.30 yd figure and has been removed, not
renumbered.) The 1.9s interval is tight enough to defend a point estimate
under `floor`; the 3s interval narrowed too (18× more tracks) but stays
wide, because 3s coverage (3.5%) is a property of how long ball flights
last, not of sample size — more weeks won't fix it.

That's the sentence that ties both tiers together, strengthened rather
than weakened by the later measurement: **Tier 0b found filters useless
on the published stream because the vendor had already filtered it —
confirmed directly, not just inferred, once this project measured that
stream and found it quantized to 0.01 yd with no detectable noise above
that. Tier 1 finds that filtering becomes necessary once *any* noise is
added back on top.** What fraction of the real sensor's own noise that
added σ actually represents is not something this data can answer — but
the finding that the published stream has already had its noise removed,
not just its velocity/direction fields, is now evidenced, not assumed.

**Finding 2 — destination value is itself horizon-dependent, which
rewrites the Tier 0a headline above.** `constant_velocity` minus
`velocity_turning_to_ball` at σ=0 (both read the same NGS velocity;
the gap is purely the landing point):

| | value of destination info | coverage |
|---|---|---|
| 1.9s (primary) | **0.31 yd** | 53.0% |
| 3.0s (secondary) | **1.51 yd** | 3.5% |
| *old headline (week 1, 3s)* | *0.887 yd* | *4.6%* |

Mechanism: at short horizons the straight-line constant-velocity
assumption hasn't broken yet — there's been too little time for a
receiver's actual route to diverge from "current heading, held" — so
knowing the endpoint buys almost nothing. It buys a lot once the flight
lasts long enough for paths to actually diverge, which is also precisely
the region with the least data. **"Destination is worth ~0.9 yd" was
never wrong, but it was a 3s-only, 4.6%-coverage number reported as if
it were the general answer; the general answer is that its value grows
with horizon, from near-zero to over a yard.**

The separate, stated prediction that `toward_ball_land` would be the
most noise-robust method across the whole range held only below σ≈0.15
at 1.9s — direction is robust to position noise, magnitude isn't, and
`kalman`'s smoothing eventually wins on magnitude despite starting
worse. Reported at the time as a partial confirmation; unchanged by the
horizon correction.

**Finding 3 — IMM under degradation: phenomenon real, mechanism still
unknown, now on much stronger evidence.** At σ=5, primary horizon: `imm`
(25.68 yd) beats `kalman_ca` (40.62 yd) by 58%, a larger margin than the
original week-1, 3s estimate (39.37 vs. 59.57, 51%) suggested. The
tested hypothesis — that IMM detects the double-differenced acceleration
state becoming useless and down-weights it toward something CV-like —
stays **falsified**. Mean P(manoeuvring) rises monotonically with σ,
re-verified on the full 18-week grid with std ≤ 0.0011 (tight enough
that this is no longer a noisy read):

| σ | 0.0 | 0.2 | 0.5 | 1.0 | 2.0 | 5.0 |
|---|---|---|---|---|---|---|
| mean P(manoeuvring) | 0.044 | 0.189 | 0.255 | 0.324 | 0.403 | 0.496 |

The **untested** alternative mechanism stands as before: at extreme σ,
`R` dominates both models' innovation covariance almost equally
regardless of their 10×-different `Q`, so their likelihoods stop
discriminating and `mu` drifts toward an even mix — IMM behaving as an
implicit ensemble average, a form of regularisation `kalman_ca` (locked
to one `Q`) cannot replicate. Still a hypothesis, not a finding.

**Finding 4 — the hybrid's condition, stated precisely.** At the primary
horizon, clean data, under the primary (`quadrature`) convention:
`tbl_kalman_speed` (5.25) *loses* to `constant_velocity` (4.57) by a
wider margin than first reported (0.68 yd, not 0.21) — "beats every
baseline" is false as originally stated, and more clearly false under
the physically-correct `R`. It crosses below `constant_velocity` at
**σ≈0.075** (Finding 1's headline number above) — comfortably before
NGS's own floor (≈2× margin) either way `R` is built. Under `floor`
(sensitivity), the crossing is earlier still (σ≈0.028) with a wider
apparent margin (≈5×) — both conventions agree on the qualitative
result, just not the exact size of the safety margin.

The rest of this finding's detail — the σ=0.30/σ=5.0 endpoint numbers
and the degradation ratios below — comes from the detailed sigma-by-sigma
grids, which are reported under `floor` throughout (sensitivity
convention, not re-run under `quadrature`; see the methodology note
below). Past the crossing the gap opens fast: at σ=0.30,
`tbl_kalman_speed`=5.36 vs. `constant_velocity`=11.31; at σ=5.0 (pure
`sigma_player`), 8.36 vs. 172.23. Degradation ratio (σ=5/σ=0):
`constant_velocity` 37.7×, `tbl_kalman_speed` 1.75× — the flattest of
every predictor compared, matching the original "~1.8×" estimate
closely. (The two `R` conventions converge well before σ=0.30, per the
headline table's note above, so these mid-to-high-σ figures are not
meaningfully convention-sensitive even though they were computed under
`floor`.)

Mean signed along-track error explains the mechanism: `toward_ball_land`/
`tbl_kalman_pos` flip from a small undershoot to a **growing overshoot**
as σ grows — their raw, finite-differenced speed inflates fast enough to
saturate the "never overshoot the landing point" cap, collapsing the
prediction to "already there" for players who, in reality, hadn't
arrived yet. `tbl_kalman_speed`/`tbl_kalman_both` instead carry a
modest, slowly growing *undershoot* — the anticipated Kalman-speed lag
bias — that never blows up.

This was a stated prediction, made before running it, against a track
record where four of the previous five predictions in this project came
back wrong or half-wrong. Confirmed for two of the three hybrid cells —
but conditionally, not unconditionally: **true past σ≈0.075
(`quadrature`, `derived` mode) — false at true zero noise, by a larger
margin than first reported.**

**Does the immunity survive an imprecise landing point? Yes, and the
sigma_player/sigma_ball comparison is a regime, not a single ratio.**
Marginal sensitivity (finite differences, primary horizon):

| predictor | d(err)/d(σ_player) @ (0.5,1.0) | d(err)/d(σ_ball) @ (0.5,1.0) | ratio | @ (1.0,2.0) player | @ (1.0,2.0) ball | ratio |
|---|---|---|---|---|---|---|
| `toward_ball_land` | 4.95 | 0.15 | 32× | 1.83 | 0.29 | 6× |
| `tbl_kalman_speed` | 1.56 | 0.09 | 17× | 0.95 | 0.10 | 9× |
| `tbl_kalman_both` | 2.23 | 0.08 | 28× | 1.35 | 0.09 | 16× |

`sigma_player` dominates everywhere tested (17–43× at the low-noise
operating point, 6–16× at the higher one — checking the low end, as the
original prediction asked, found it *further* from parity, not closer).
But the ratio **shrinks as noise grows**, because `sigma_player`'s slope
saturates (physical displacement can only degrade a bounded quantity so
much per step) while `sigma_ball`'s keeps rising. Report it as a regime:
**at low noise levels, player-position quality is what matters; only
under heavy, compounding degradation do the two axes begin to converge.**

Two coupled scenarios (stated assumptions about how the axes might track
together, not measurements of a real relationship) both show the
hybrid's advantage surviving: along `σ_ball = σ_player`, `tbl_kalman_speed`
reaches 9.19 yd at σ=5 vs. 13.04/20.00 for `toward_ball_land`/
`velocity_turning_to_ball`; along the harsher `σ_ball = 3·σ_player`,
7.88 vs. 13.25/13.58 at σ_player=2.

`toward_ball_land` and `tbl_kalman_speed` share their direction
computation, so they should degrade at about the same rate along the
pure `sigma_ball` axis — confirmed at both horizons: slopes 0.191 vs.
0.202 yd/yd at 1.9s, 0.218 vs. 0.231 at 3s. **Both well under 1 yd of
error per yd of landing-point noise — destination error propagates
benignly**, regardless of which destination-based method is used. The
σ_player-axis saturation failure mode from Finding 4 does not reappear
here.

**Net effect on the engineering recommendation**: weaker than the
motivating concern suggested, not stronger, and now precisely bounded.
Because per-unit sensitivity favours `sigma_player` except under heavy
degradation, the sharp claim ("stop spending on localisation, spend on
landing-point estimation") isn't supported at realistic operating
points — it would need real-world `sigma_ball` to be many times larger
than `sigma_player` in absolute terms, which this project has no
measurement of. What holds: the hybrid's near-immunity to *player*
localisation noise is real, survives realistic landing-point error, and
its landing-point sensitivity is inherited unchanged from plain
`toward_ball_land` — not a new weakness the hybrid design introduced.

Honesty constraints for this axis, re-verified at full scale: `σ_ball=0`
collapses `corrupt_input_two_axis` onto plain `corrupt_input` exactly
(not approximately) for every `(σ_player, seed)` checked; the assertion
that non-destination predictors (`constant_velocity`, `kalman`,
`kalman_velocity`, `imm`, `kalman_ca`) are invariant to `σ_ball` was
proven both ways — fires on a deliberately broken corruption path,
passes on the correct one — then held at every real grid point tested.

### Methodology note: measuring the sensor's actual noise floor

This project used three different figures for "NGS's accuracy" across its
lifetime — a config comment citing "NGS/Zebra's public figure, roughly 6
inches (~0.17 yd)"; a design constant, `KALMAN_R_POS`, σ≈0.25 yd, chosen
deliberately more conservative than that figure; and a "published range,
0.15–0.30 yd" quoted throughout this document and README.md as NGS's own
stated accuracy. Checking the primary source
([operations.nfl.com/gameday/technology/nfl-next-gen-stats](https://operations.nfl.com/gameday/technology/nfl-next-gen-stats)):
NFL Football Operations states only that the system *"charts individual
movements within inches"* — no numeric figure. **The 0.15–0.30 yd range was
fabricated and has been removed from every file it appeared in, not
renumbered.** The 0.17 yd config comment is itself uncited background, not a
verified vendor figure.

Rather than replace one assumption with another, this project measured the
published position stream directly.

**Attempt 1 (abandoned): stationary pre-snap players.** The idea — many
players are motionless before the snap, so their reported position's
variation is pure measurement noise — requires selecting frames on game
situation only, never on position or speed (that would bias the estimate
downward). This dataset has no event column at all (checked: the full
23-column input schema has no `line_set`/`ball_snap`/motion marker) and no
offensive line positions either (`player_position`'s full value set is
`{CB,DE,DT,FB,FS,ILB,MLB,NT,OLB,QB,RB,S,SS,TE,WR}` — this competition's data
excludes linemen). Used `player_role=="Passer"` (the QB) and the earliest
frames of each play as the best available substitute, swept window length
5 to 25 frames. Both a successive-difference and a detrended-residual
estimator were run; they disagreed by 20-170% depending on window length,
and grew with window length in both directions — small windows biased the
detrended estimator toward zero (fitting a line through 3-5 points
absorbs real variation into the trend), large windows picked up genuine
motion (the autocorrelation of residuals at 15-25 frames showed the
smooth, slowly-decaying signature of unremoved curvature, not noise).
**Abandoned**: there is no window in this dataset's tracked history that is
both long enough to detrend reliably and short enough to still be
stationary.

**Attempt 2: difference-order ladder, no stationarity assumed.** Real
motion is smooth; sensor noise is white; differencing is a high-pass
filter. For white noise of variance σ², the n-th order difference has
variance `C(2n,n)·σ²` (verified: the sum of squared binomial coefficients
of order n equals the central binomial coefficient of order 2n — n=1→2,
n=2→6, n=3→20, n=4→70, matching the claimed divisors exactly), so
`σ̂(n) = std(nth difference) / sqrt(C(2n,n))`. No window, no role filter,
no stationarity — every frame of every play. Validated twice before
trusting it on real data: on pure synthetic white noise (σ=0.10 recovered
as 0.100-0.101 at every order) and on synthetic sinusoidal motion plus
σ=0.10 noise (order 1 badly biased at +70%, order 2 at +2.9%, orders 3-4
within 0.3% — the predicted convergence pattern). Run on real data,
training games, every tracked frame:

| order | x | y | pooled |
|---|---|---|---|
| 1 | 0.2124 | 0.1629 | 0.1892 |
| 2 | 0.0113 | 0.0101 | 0.0107 |
| 3 | 0.0046 | 0.0042 | 0.0044 |
| 4 | 0.0044 | 0.0040 | **0.0042** |

Order 1→2 drops 94%, order 2→3 drops 59% (still motion-contaminated by the
stop condition specified for this check — orders 2, 3 and 4 were required
to mutually converge). Order 3→4 drops 5.1% — close, not a clean pass.
Test-games order-4 figure: 0.0038 (consistency check, no decision attached).

**Attempt 3: power spectrum.** Averaged periodogram (per-track linear
detrend, Hann window, interpolated onto a common frequency grid, averaged
across tracks) rather than one long Welch average, since thousands of
independent short plays stand in for that. Initial 3-5Hz floor (x=0.0043,
y=0.0039) showed an apparent decline from 4.2 to 5.0Hz; re-ran the
identical pipeline on synthetic white noise with the same track-length
distribution and found the same decline — a windowing/interpolation
artifact at the Nyquist edge, not evidence of smoothing. Floor recomputed
on the artifact-free 3.0-4.2Hz band: **x=0.0046, y=0.0041**, confirmed flat
there in both real and synthetic data.

**The two methods agree (within ~5%) at ≈0.004-0.005 yd — but this is not
independent confirmation of a sensor-noise measurement.** Both methods
differ only in how they extract high-frequency content; both assume that
content is sensor noise. That assumption is what failed, not the
extraction. Checked directly:

1. **Decimal precision**: `x` and `y` in the raw files are exact multiples
   of 0.01 yd, no exceptions (multiplying by 100 gives an integer 100.00%
   of the time).
2. **Difference histogram** (the decisive test): **100.00%** of `x[t]-x[t-1]`
   values land exactly on the 0.01 grid. Discrete spikes at 0, ±0.01,
   ±0.02..., nothing in between. (A figure of this histogram existed at
   one point but its generating script is gone and could not be recovered
   from git history or the working tree; not re-citing an artifact that
   can't be reproduced — the claim itself is a committed-data fact,
   checkable directly against `data/train/`.)
3. **End-to-end validation of the estimator itself**: injected known σ=0.10
   noise into the *real* (unquantized) motion tracks and re-ran the ladder
   — recovered 0.1007, 0.1001, 0.1001 at orders 2/3/4, clean convergence by
   order 2. The tool correctly detects real additive noise; it does not
   manufacture a false floor.

0.0042 yd (3.7 mm) is not a plausible UWB measurement — the best published
UWB hardware is on the order of 10 cm, and Zebra's own uncited figure above
is ~15 cm. A system with inches-level real error cannot produce a published
stream whose residual noise is undetectable above a 3.7 mm quantization
step **unless the noise was removed before publication.** The quantization
step itself did not create 15x the noise floor it represents: the
difference-order ladder recovers injected noise 25x larger than this
quantization step almost exactly (item 3 above), so quantization at 0.01 yd
cannot hide something 15x its own size. **The white noise was not hidden.
It was removed before publication.**

**Finding, stated plainly: the published NGS position stream is
quantized to 0.01 yd and carries no detectable white noise above that
step. Raw sensor noise is not recoverable from this data.** This
strengthens, not weakens, this project's opening observation that NGS's
`s`/`dir` columns are vendor-derived, not raw measurements (see Tier 0b
above) — extended now to position itself: nothing in this dataset is a
raw measurement. The NFL's own public description of the system — 20-30
receivers per venue, 2-3 tags per player at 10 Hz — makes this unsurprising
in hindsight: fusing multiple tags into one coordinate is itself a
processing step, before whatever smoothing is applied on top.

**Consequence for every number in this document and README.md expressed
relative to a sensor-accuracy figure**: the comparison has been removed,
not renumbered, wherever it appeared (Finding 1's two crossings, the
Tier 0b/Tier 1 tie-together sentence, the limitations note on `kalman`'s
crossing). The measured values themselves — 0.075, 0.147, 0.1432, 0.9338 —
are unaffected and stand as measurements. What any of them means relative
to the real sensor's own noise is now known to be unanswerable from this
data, not approximately 2x or 3.7x or break-even.

### Methodology note: the coverage guard that only guarded the plots

3 s was this project's headline horizon from Tier 0a onward, chosen
because a median ball flight is about 1.9 s and 3 s felt like a
reasonable stretch target — without checking how many tracks actually
survive that long. They don't: at 3 s, 3.5% of tracks remain (394 of
11,254 on the full dataset; it was 4.6%, 24 of 526, on week 1 alone).
`config.MIN_COVERAGE = 0.25` existed from the start specifically to
guard against exactly this, and it worked — for `plot_error_curves`,
which stops drawing a line once its coverage drops too low. **It was
never wired into a single table.** Every headline number in this
project through Part 3 of Tier 1, including the original "6.40 yd @3s"
bar, was a bare number with no coverage attached, because the guard
that would have caught it only fired on a chart nobody was reading
numbers off of. **A guard that only fires on plots is not a guard.**

The fix (`scripts/run_tier1_taskn.py`): measure coverage properly across the
full dataset at every horizon step, and choose the primary horizon
data-driven — the largest step at ≥50% coverage, which lands at 1.9s
(53.0%), almost exactly where the median-flight-time guess would have
put it, but now with a number to back it rather than an assumption.
`evaluate.summarise()` and the new `evaluate.summarise_at_step()` attach
`(n, coverage)` to every horizon column they produce; `3s` is kept as a
secondary metric, always printed with its 3.5% coverage next to it,
never bare. Every claim in this document was then re-run on the full
18-week dataset at both horizons before being restated above.

**On the `R`-convention decision (adopted early in this section) and the
discrepancy it resolves.** An earlier coverage-audit script reported
`tbl_kalman_speed` @1.9s, σ=0 as 5.25 yd; the sigma-by-sigma sweep
tables in this section, still run under `floor`, report 4.77. Both were
correct for what they measured — `run_tier1_sweep.set_filter_R`'s
`σ_pos = max(σ, 0.05)` floor (added in Part 2 to keep `R` invertible at
σ=0) gives every Kalman-family predictor's "σ=0" row `R_POS = 0.05²`,
not Tier 0b's UWB-sourced `0.25²`. Verified by forcing each `R`
directly: Tier 0b's reproduces 5.25 exactly, the floor reproduces 4.77
exactly. Adopting `quadrature` as primary (reasoning above) resolves
this as a side effect rather than a deliberate fix: `quadrature`'s σ=0
value *is* `KALMAN_R_POS` by construction, so the standalone Tier 0b
number and the sweep's σ=0 number stop being two different numbers.

Per-predictor, the `floor`→`quadrature` clean-data delta (1.9s) grows
with how much of a predictor's estimate comes from trusting position
measurements directly — smallest for `tbl_kalman_pos` (+0.14 yd, only
the anchor uses the Kalman posterior), largest for plain `kalman`
(+1.28 yd, everything it outputs comes from trusting position
measurements). Full table, both conventions, both horizons:
`outputs/tier1_r_convention_clean_table.csv`. (The `output_*.csv` "truth"
every number here is scored against is itself NGS-measured, not
physical reality — see the last Limitation below.)

**The `R` correction, including the part that looks bad.** `R` is now
block diagonal (`diag(σ_pos², σ_pos², σ_vel², σ_vel²)`) rather than one
isotropic value scaled by σ. `σ_pos = max(σ, 0.05)`, unchanged. `σ_vel`
in `derived` mode is `sqrt(2)·σ_pos/dt` — the analytically correct noise
on a one-step finite difference of two noisy positions, not σ itself.
In `ngs` mode, `σ_vel` is a constant, 0.3 yd/s — an assumption, since
NGS publishes no velocity accuracy spec, independent of σ because the
velocity columns are never corrupted in that mode.

The old, mis-specified `R` understated `σ_vel` by roughly 14× in
`derived` mode. Correcting it makes some numbers *worse*:
`kalman_velocity`'s σ=5 error went from **24.35 to 30.13 yd**. The
pre-fix number was better and was wrong — it came from a filter
over-trusting a velocity "measurement" it should have almost entirely
ignored. The corrected number is what's reported everywhere else in
this document.

The same fix moved the two modes in *opposite* directions, which is the
actual evidence it was the right fix, not just a different wrong one. In
`derived` mode, the "measurement" is a deterministic function of
positions the filter already has, so a correctly specified filter gives
it almost no weight — `kalman_velocity` converges onto plain `kalman`'s
numbers. In `ngs` mode the velocity really is an independent, clean
signal, and there `kalman_velocity` separates and finally beats
`constant_velocity` at σ=5 (8.75 vs. 10.33 yd) — it was *losing* to
`constant_velocity` at every σ before the fix, which never made physical
sense for a channel that was never corrupted.

**Theory cross-check.** For `constant_velocity` in `derived` mode,
`MDE_noise ≈ 53.2·σ` (closed form: √2 from differencing two independent
samples, ÷dt for the interval, ×3 for the 3 s horizon, ×1.253 for the
mean of a 2-D Rayleigh displacement), combined in quadrature with the
σ=0 floor:

| σ | predicted | measured | ratio |
|---|---|---|---|
| 0.00 | 7.318 | 7.318 | 1.000 |
| 0.05 | 7.786 | 8.223 | 1.056 |
| 0.10 | 9.045 | 9.880 | 1.092 |
| 0.20 | 12.907 | 14.545 | 1.127 |
| 0.30 | 17.547 | 19.801 | 1.128 |
| 1.00 | 53.662 | 59.754 | 1.114 |
| 2.00 | 106.572 | 117.972 | 1.107 |
| 5.00 | 265.902 | 293.124 | 1.102 |

The ratio holds at **1.10–1.13 across two decades of σ** (0.05 to 5.0).
A constant ratio is the important result: it means the closed form's σ¹
scaling is correct — an unmodelled *additive* error source would make
the ratio fall as σ grows, and it does not. A roughly 12% multiplicative
offset remains unattributed. One candidate — noise in the anchor
position itself, which `constant_velocity` extrapolates from — was
checked numerically and accounts for only ~1.8% of the gap, not 12%,
and, being additive, has the wrong shape to explain a ratio this flat.
The offset is reported as open, not force-fit to an explanation that
doesn't hold up. (This table is at 3s specifically, 3.5% coverage; not
yet recomputed at the 1.9s primary horizon.)

**Limitations:**

- Finite-differenced velocities at consecutive frames share a position
  sample (`v[t]` and `v[t+1]` both use `x[t]`), so they're correlated —
  violating the white-measurement-noise assumption the Kalman update
  makes. Accepted approximation, not corrected.
- 3s coverage (3.5%) is inherent to how long ball flights last, not a
  sample-size artefact — more weeks narrow the 1.9s primary-horizon
  interval effectively (18× the data took the bootstrap width from
  0.064 to 0.0066) but cannot fix 3s the same way, since the fraction of
  throws that stay airborne that long doesn't change with more weeks of
  the same kind of football.
- `ngs` mode's velocity σ (0.3 yd/s) is a stated assumption, not a
  published spec.
- The landing-point noise model (Finding 4) is a fixed isotropic
  Gaussian. A real landing-point estimator's error is neither isotropic
  nor constant — it shrinks as the ball nears landing and is larger
  along the flight direction than across it. Recorded here rather than
  fixed, since modelling that properly needs a model of the estimator,
  not just of its error magnitude.
- The two coupled scenarios in Finding 4 (`σ_ball = σ_player` and
  `σ_ball = 3·σ_player`) are stated assumptions about how the two noise
  sources might track together in a real deployment, not a measured
  relationship — the true coupling, since both derive from the same
  underlying sensor, is unknown.
- An earlier version of this document compared `kalman`'s crossing to
  "the edge of NGS's accuracy," using an assumed published range (0.15–0.3
  yd) that turned out to have no source (see the Methodology note on the
  sensor-noise measurement). That comparison has been removed rather than
  reworded. The crossing itself (0.147 under `quadrature`, a 2-seed check,
  not the full bootstrap) stands as a measurement with no claim about what
  it means relative to the sensor.
- The sigma-by-sigma detail tables in Findings 2–4 (everything past the
  σ=0 clean-data numbers and the two headline crossings) are reported
  under `floor`, the sensitivity convention, not `quadrature`, the
  adopted primary one — re-running them was judged not to change the
  qualitative results (the two conventions converge above roughly
  σ≈0.3, verified numerically for σ=5) but they have not been
  individually re-verified point-by-point under `quadrature`.
- `output_*.csv` — the ground truth every number in this project is
  scored against — is itself NGS's measured output, not physical
  reality. Physical truth appears nowhere in this dataset; input and
  output carry the same underlying measurement process. This doesn't
  invalidate relative comparisons between predictors (all scored against
  the same measured "truth"), but "error" throughout this document has
  always meant distance from NGS's measurement, not from where a player
  physically was.
- The `R`-floor discrepancy (methodology note above) means every
  Kalman-family predictor's degradation-sweep "σ=0" row is not the same
  filter as Tier 0b's clean-baseline table. Documented, not corrected —
  correcting it would mean re-running every sweep in this section again.

## Design notes

**Play direction is normalised.** Leftward plays are mirrored so every play
runs in `+x`. The mirroring has to reach the output file and the ball
landing point too — the output file carries no `play_direction` column of
its own, so the flag is joined from the input. Missing either produces a
pipeline that looks healthy and scores nonsense.

**Acceleration is projected onto the current heading.** NGS gives `a` as a
magnitude with no direction. Treating it as along-track is exact when a
player accelerates in a straight line and wrong when they cut — which is
precisely why the constant-acceleration baseline fails on turns.

**Every predictor shares one signature** —
`f(sample, horizon) -> (n_players, horizon, 2)`, where `sample` is a
`data_loader.PlaySample`. The six baselines above only ever touch
`sample.state` (the anchor frame); Kalman and IMM are the first predictors
that also read `sample.history` (every pre-throw frame per player). The
signature carries the whole sample, rather than just the state frame, so
that dependency is visible in the function itself instead of hidden behind
a side channel.
