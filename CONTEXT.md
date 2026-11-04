# Project context

Handoff notes: why this project exists, what has been decided, and what is
left. `README.md` covers the technical setup; this file covers the reasoning
behind it, which nothing in the code records.

If you are an assistant picking this up cold, read this file, then
`README.md`, then `config.py` and `data_loader.py`.

A note on the run logs below: they record commands as they were actually
typed at the time (e.g. `python run_tier0a.py`), before the repo was
reorganised into a `scripts/` directory. Run scripts now live in
`scripts/`; library modules (`config.py`, `data_loader.py`, ...) are
unchanged at the repo root. See README.md's Reproducing section for the
current invocation.

---

## What this is for

A portfolio project, not a competition entry.

Roy is an incoming ECE master's student at the University of Toronto,
specialising in communications AI and signal processing (wireless: Wi-Fi /
Bluetooth). Undergrad was statistics with maths and CS minors. The target
employers are two groups that normally want different things:

- **Wireless / silicon** — Ericsson Canada, Nokia Bell Labs, Qualcomm
  Canada, Realtek's wireless division on a return to Taiwan.
- **Large tech / FAANG-level**, in Canada after graduation.

The competition itself is not a goal. A high leaderboard placement is not a
goal. Working in sports analytics is explicitly **not** a goal.

## The framing, and why it is this one

The common ground between those two groups is one problem statement:

> Real-time trajectory prediction under constrained sensing and a hard
> latency budget.

Three deliverables read differently to each audience while being the same
work:

| Deliverable | Large tech reads | Wireless reads |
|---|---|---|
| accuracy vs latency budget | real-time serving, SLO design | edge inference feasibility |
| accuracy vs input degradation | robustness, failure modes | sensor spec vs algorithm trade-off |
| accuracy vs model size | cost/performance tuning | on-device memory and power limits |

One codebase, one set of experiments. Only the order of presentation
changes.

The resume line this is built to support, which works unmodified for either
audience — note it contains no football terminology:

> Built an end-to-end streaming trajectory-prediction system on 10 Hz
> multi-target UWB tracking data (NFL Next Gen Stats); benchmarked
> model-based Kalman/IMM filters against learned sequence models,
> characterised accuracy degradation under injected sensor noise, dropout
> and reduced update rates, and deployed quantised inference on AWS with
> measured p99 latency.

### On the UWB angle

Next Gen Stats positions come from UWB tags in players' shoulder pads, read
by stadium receivers at ~10 Hz. That is a fact about the data, not a
rhetorical device.

But the *motivation* is separate from the *content*, and conflating them
produces something Roy cannot say honestly in an interview. The honest
version: he is a football fan who wanted a project with real tracking data;
what makes it technically interesting is that the data is a UWB
localisation system's output, so it can be treated as a state-estimation
problem with a sensing-degradation study attached.

Roy had not encountered UWB before this project started. If the framing
ever starts to feel like a stretch, the fallback is to keep every
experiment and simply not invoke UWB — "model-based filters vs learned
sequence models for multi-target trajectory prediction" stands on its own.
The code does not change either way. This decision can be deferred
indefinitely.

## Tiers, effort, and where to stop

Each tier is a valid stopping point. Nothing is left half-finished if the
term gets busy.

| Tier | Content | Rough effort | Status |
|---|---|---|---|
| 0a | Pipeline, physics baselines, animation | ~8 h | done; results in README |
| 0b | Kalman filter, IMM, state estimation | ~12 h | done; results below and in README — loses to `constant_velocity`, narrowed by IMM but doesn't beat it |
| 1 | Learned model + degradation study | ~30 h | Part 1 (degradation sweep) complete — every claim verified at full scale (18 weeks) at both the 1.9s primary and 3s secondary horizon. Learned-model part (Task K) not started; Task P's CV protocol comes first |
| 2 | Streaming, quantisation, latency, AWS | ~25 h | not started |

**Tier 1 is the value centre.** The resume line above becomes true at the
end of Tier 1; Tier 2 is upside. Priority when time is short is
B > A > C in the original framing: the degradation study matters more than
the filter, which matters more than the cloud work.

Tier 0a must not be allowed to consume the term. Its job is to establish a
floor, not to be good.

### Why 0a and 0b are split

Roy has not worked with a Kalman filter before. Tier 0a therefore uses
constant-velocity extrapolation — five lines, no theory — so that the
pipeline is proven end to end before any estimation theory is introduced.
Tier 0b then swaps in the filter against a working pipeline, and the error
drop from one to the other is itself a teaching moment and a README figure.

Do not collapse these two steps.

## Decisions already taken

**Scope is deliberately narrow.** Long ball flights only
(`MIN_HORIZON = 15` frames, `MAX_HORIZON = 40` in `config.py`), and
`WEEKS = ["w01"]` by default — one week, exactly as this file always said,
raised to more weeks only once you know how long a run takes. Full-season
coverage is for leaderboard chasers. A clean comparison beats a broad one
here, and it keeps everything on a laptop.

**Compute is unresolved and does not need resolving.** Tier 0 and most of
Tier 1 run on a laptop given the narrow subset. Tier 2 is where cloud is
needed, and by then the question will answer itself.

**Football knowledge is not required and is not a weakness.** Roy watches
the game but does not know scheme detail (routes, coverages). The chosen
framing does not depend on it — what is needed from the data is geometry:
who is targeted, who is covering, how long the ball is in the air. All
three are columns.

**The 2026 competition data is the working set**, not 2027. The eighth
annual competition ran Sept 2025 – Jan 2026 and is closed; its data,
solutions and discussion are all public, which is worth more than a live
leaderboard. The ninth (2027) had not been announced as of 2026-09-20 and
is expected imminently. When it opens, this pipeline should transfer with
minor schema edits.

## Things that were considered and rejected

**Pure ML-systems / cloud framing (no sensing angle).** Effective for large
tech, close to worthless for wireless employers, and it collapses back into
"a model-training project", which is the thing Roy explicitly did not want.

**Chasing leaderboard rank.** Wrong objective. A good score adds nothing to
either target audience and consumes all the time that the degradation study
needs.

**Writing two differently-framed versions of the project.** Unnecessary —
one set of experiments supports both readings.

**A coverage guard that only gates plots, not tables.**
`config.MIN_COVERAGE = 0.25` existed from Tier 0a onward and was wired
into `plot_error_curves`, which correctly stops drawing a predictor's
line once its coverage drops too low. It was never wired into any
table-producing function. The result: every headline number through
Tier 1 Part 3, including the original "6.40 yd @3s" bar, was reported
bare, and 3s coverage turns out to be 3.5% on the full dataset (4.6% on
week 1) — a number nobody would have quoted as a headline if it had been
printed next to the metric. Rejected as a practice going forward:
`evaluate.summarise()` and `evaluate.summarise_at_step()` now attach
`(n, coverage)` to every horizon column they produce, and no number in
README.md is reported without one. See README.md, "Methodology note: the
coverage guard that only guarded the plots," and `run_tier1_taskn.py`
for the audit that caught it and picked the corrected primary horizon
(1.9s, the largest step at ≥50% coverage). The lesson generalises past
this project: a guard proven correct in one code path is not proven
correct everywhere that path's result gets used.

## Current state

Tier 0a has been run end to end on the real Kaggle files
(`python run_tier0a.py`, 2026-09-20; now `scripts/run_tier0a.py` after the
scripts/ reorganisation), on week 1 only (`WEEKS = ["w01"]`,
the default). The pipeline now anchors each play at the real throw frame
(`player_to_predict`) and uses the play's actual `num_frames_output` as its
horizon, rather than a fixed lookback — 128 plays, 526 player-tracks,
horizon 15–34 frames (median 1.9 s).

Mean displacement error (yards), lower is better:

| predictor | overall | @1s | @2s | @3s |
|---|---|---|---|---|
| constant_velocity | 2.256 | 1.693 | 4.776 | 7.284 |
| velocity_turning_to_ball | 2.339 | 2.070 | 4.608 | **6.397** |
| toward_ball_land | 2.876 | 2.769 | 4.883 | 6.498 |
| constant_acceleration | 2.952 | 1.980 | 6.439 | 11.101 |
| damped_velocity | 3.678 | 2.588 | 8.249 | 14.670 |
| constant_position | 6.908 | 5.896 | 13.277 | 20.243 |

By role (mean error, yards):

| predictor | Defensive Coverage | Targeted Receiver |
|---|---|---|
| velocity_turning_to_ball | 2.707 | 1.174 |
| constant_velocity | 2.340 | 1.991 |
| toward_ball_land | 3.310 | 1.499 |
| constant_acceleration | 2.921 | 3.048 |
| damped_velocity | 3.565 | 4.033 |
| constant_position | 6.391 | 8.545 |

Readings:

- **`constant_velocity` wins overall and through ~2s, but `toward_ball_land`
  alone already beats it by @3s** (6.498 vs 7.284) — no blending, no
  heading, just walking straight at the known landing spot. That is the
  actual finding, not "the blend works": destination geometry alone
  becomes the dominant signal at the long horizon, and only trails early
  (through ~2s) because it ignores current heading entirely.
  `velocity_turning_to_ball` blends in `constant_velocity` to patch that
  early gap and edges a little further ahead (6.397 at 3s), but the win
  belongs to the landing-point information, not the blending trick. Get
  this backwards and it reads as a tuning anecdote instead of a fact about
  the problem's structure — which is exactly what it should set: **Tier 1's
  bar at 3s is 6.40 yd, not `constant_velocity`'s 7.28**, and most of that
  6.40 comes from geometry the model doesn't need to learn. `toward_ball_land`
  is also visibly bounded (can't overshoot), unlike `constant_acceleration`
  and `damped_velocity`, which diverge past ~1.5s.
- **The role split inverts the hypothesis this file used to have.**
  `Targeted Receiver` error is *lower* than `Defensive Coverage`, not
  higher, for every baseline except `constant_acceleration` and
  `damped_velocity` [corrected 2026-11-03: `constant_position` inverts as
  well]. Giving every player the same known landing point
  turns out to be far more informative for the receiver actually running
  to it than for the defender reacting in real time — worth stating
  plainly in the README rather than assuming intent is harder to predict
  on offence.
- `constant_acceleration` and `damped_velocity` still diverge badly past
  ~1.5s, same qualitative failure as before, now sharper on this smaller,
  real-anchor sample (`damped_velocity` reaches 14.7 yd @3s).
- Figures in `outputs/`: `error_curves.png`, `error_by_role.csv`,
  `baseline_errors.csv`, `play_animation.gif`.

Environment: a project `.venv` (Python 3.14.7, pandas 3.0.6, numpy 2.5.3,
matplotlib 3.11.2), not the conda `base` env an earlier version of this
file mentioned — use `.venv`. Still worth creating a `python=3.12` env
before Tier 1 brings PyTorch in, since 3.14 support is still patchy.

### Tier 0b (Kalman / IMM), run 2026-09-22

Same week-1 dataset as Tier 0a (128 plays, 526 player-tracks), via
`python run_tier0a.py` (now `scripts/run_tier0a.py`).
`data_loader.PlaySample` now carries a `history`
field (every pre-throw frame per player, not just the anchor), and every
baseline's signature changed from `f(state, horizon)` to
`f(sample, horizon)` — a correction to my own spec mid-round, which had
tried to smuggle history through `state.attrs` instead of just changing
the signature honestly. Verified byte-identical against the pre-change
output before adding anything Kalman-related.

**Single-model Kalman (`kalman.py`) loses to `constant_velocity` at every
horizon**, both with `H` observing position only and with `H` also
observing NGS's reported velocity:

| predictor | overall | @1s | @2s | @3s |
|---|---|---|---|---|
| constant_velocity | 2.256 | 1.693 | 4.776 | 7.284 |
| kalman_velocity | 2.883 | 2.342 | 5.738 | 7.715 |
| kalman | 3.720 | 3.182 | 6.959 | 8.616 |

Real finding per `TIER0B_SPEC.md`, not tuned away. `kalman_velocity`
beating plain `kalman` at every horizon is itself informative: trusting
NGS's own reported velocity as a measurement helps more than deriving
velocity purely by differencing position — the opposite of the
"differencing amplifies noise" hypothesis this project started with.

**Diagnosed with NIS, not by tuning against the result.** Differencing
NGS's own velocity gives `sigma_a = 2.388 yd/s²`, known biased high
(differencing amplifies noise). Its NIS is 1.27 against an expected 2 —
confirms `Q` is too large. Bisected `sigma_a` down to `1.791`, where NIS
lands at 1.99/2.87 for the two `H` configurations (answer-blind: only the
filter's own priors and measurements, and the NIS computation itself was
validated first against a synthetic model where the answer is known,
landing at NIS ≈ 2.00/4.00 there, before trusting it on real data).

**The NIS-consistent `sigma_a` makes prediction accuracy *worse*, not
better** (`kalman` @1s: 2.986 → 3.182 yd, raw vs. calibrated). This was
the useful surprise of the round: NIS consistency and prediction accuracy
are different objectives under model mismatch. A larger `Q` gives a
larger Kalman gain, so the filter leans on each new noisy position more
and its own constant-velocity prediction less — noisier but responsive
to real velocity changes. Shrinking `Q` to be "correct" makes the filter
trust its own stale prediction more, which is wrong exactly when a
receiver is cutting into the throw. This resolves the inconsistency
flagged mid-round (differencing-amplified noise vs. already-smoothed NGS
data, which pull in opposite directions): neither dominates — real,
directed player acceleration near the throw does, and it's structural,
not a tuning problem. Used the NIS-consistent value anyway for the
reported numbers, on the principle of preferring the defensible answer
over whichever one happens to score better.

**IMM (`imm.py`) narrows the gap to `constant_velocity` almost entirely,
but not by regime-switching.** Two 6-dim constant-acceleration models
(`[x, y, vx, vy, ax, ay]`, sharing one state to avoid padding state
vectors between a 4-dim and 6-dim model on every mix step — a deliberate
simplification, documented as such in `imm.py`) differing only in
process noise (`manoeuvring` = 10× `steady`, `config.IMM_MANOEUVRE_RATIO`,
a fixed design choice, not fit). Transition matrix
`[[0.95,0.05],[0.10,0.90]]`, also fixed.

| predictor | overall | @1s | @2s | @3s |
|---|---|---|---|---|
| constant_velocity | 2.256 | 1.693 | 4.776 | 7.284 |
| imm | 2.333 | 1.770 | 4.887 | 7.383 |
| kalman_velocity | 2.883 | 2.342 | 5.738 | 7.715 |
| kalman | 3.720 | 3.182 | 6.959 | 8.616 |

Two follow-up experiments (1 h timebox), both against the observation
that the manoeuvring model's probability never crosses 0.5 across the
whole dataset (0.0% of scored frames):

- **Experiment A** — a standalone 6-dim CA Kalman filter with no mixing
  at all (`kalman_ca`, `imm.make_kalman_ca`) scores 2.334 / 7.385,
  matching `imm` to within 0.002 yd. Confirms the gain is the explicit
  acceleration state, not switching — the manoeuvring model and the
  mixing step are doing nothing measurable at these settings.
- **Experiment B** — re-specified `sigma_steady` from 1.791 (the
  single-model NIS-calibrated value, optimal for one model covering all
  behaviour, and therefore already permissive enough to absorb real
  cuts) down to 0.3 (clean model separation in `test_imm.py`'s synthetic
  case). One shot, reasoning written into
  `config.IMM_STEADY_SIGMA_A_EXPERIMENT` before running. Manoeuvring
  share moved off the floor (0.0% → 2.1%), but accuracy got *worse*
  (7.383 → 7.473 yd @3s). Kept as run, not iterated on.

**Reading it together**: every filter here is solving "estimate current
velocity well," and `constant_velocity` already sits close to that
problem's ceiling — more filtering machinery mostly can't move it,
because none of these models know where the ball is going. That's most
of what `velocity_turning_to_ball`'s 6.40 yd bar is made of. This is the
argument for Tier 1, now demonstrated rather than assumed: a learned
model doesn't need to out-filter physics, it needs to use information
physics never had.

**Ceiling conclusion, one line**: state estimation is bounded near
`constant_velocity` because the measurement's velocity content
(NGS `s`/`dir`) is already good; the remaining ~0.9 yd down to
`velocity_turning_to_ball` is destination information no filter can
reach.

That conclusion is **conditional on current data quality** — it holds on
the published NGS stream as it actually arrives. (Later measured directly,
2026-10-31: the stream is quantized to 0.01 yd with no detectable noise
above that step — see PART1_DETAIL.md's methodology note. No numeric NGS
accuracy spec is published; an earlier "σ ≈ 0.15–0.3 yd" figure cited here
was unsourced and has been removed.) Whether the conclusion survives noise
added on top of that stream is untested here; that is exactly what Tier 1's
degradation sweep (Task B, `TIER1_SPEC.md`) is for, not assumed.

New files: `kalman.py`, `imm.py`, `test_kalman.py` (6 synthetic checks —
recovers constant velocity, recovers a stationary target, the
velocity-observing variant beats position-only, the `sigma_a` estimator
recovers a known injected value, short-history fallback doesn't crash,
and the NIS computation itself is validated against a well-specified
synthetic model before being trusted on real data), `test_imm.py` (3
checks, including the required manoeuvre-detection test — which needed
`sigma_steady = 0.3` and a sustained acceleration *ramp*, not a one-time
constant-acceleration step, to get a clean signal; a single step change
has zero jerk after the first couple of frames and doesn't sustain a
manoeuvring verdict, which is documented in the test file as a real
modelling point, not just a test-tuning detail).

New files in `outputs/`: `imm_model_probabilities.csv`,
`imm_model_probabilities_resigma.csv` (per-frame `mu_steady` /
`mu_manoeuvre`, per player-track, for both IMM variants).

### Tier 1 Part 1 (degradation sweep), run 2026-09-23/24

Ran end to end via `degrade.py`, `run_tier1_sweep.py` (coarse 0-5 yd
grid), `run_tier1_crossover.py` (fine 0-0.3 yd grid, play-level
bootstrap), `imm_divergence_check.py`. Full write-up is in README.md's
"Tier 1: degradation sweep" section; summarised here:

- **Finding 1**: the crossing sigma between `kalman` and
  `constant_velocity` is 0.029 yd, 95% CI [0.004, 0.068] — small, so
  filtering becomes necessary once not very much noise is added, even
  though Tier 0b found it useless on NGS's own, already-processed output.
  (The original note here compared this interval to "NGS's published
  accuracy, 0.15-0.3 yd"; that figure was unsourced and has been removed —
  see PART1_DETAIL.md's methodology note on the sensor-noise measurement.)
- **Finding 2**: `toward_ball_land`'s predicted noise-robustness
  (direction is geometric, not differenced) held only below σ≈0.15 —
  `kalman`/`kalman_velocity` overtake it beyond that, since single-frame
  destination geometry has no smoothing to fall back on. The prediction
  was made before running and came back half right; stated as such.
- **Finding 3**: IMM beats `kalman_ca` sharply at high σ (39.37 vs 59.57
  at σ=5) but not via the tested mechanism — P(manoeuvring) rises
  toward 0.5 as σ grows, not toward 0, falsifying the "abandons the CA
  state" hypothesis. An alternative (implicit ensemble averaging as `R`
  comes to dominate both models' likelihood equally) is stated as
  untested, not as an explanation.

**R correction**: was isotropic (`max(sigma,0.05)^2 * I`), wrongly
scaling velocity noise with position sigma in both modes. Now
block-diagonal: `sigma_vel = sqrt(2)*sigma_pos/dt` in `derived` mode
(analytically derived from the finite-difference noise propagation, not
fit), a constant 0.3 yd/s in `ngs` mode (stated assumption -- no NGS
velocity spec exists). The fix moved `kalman_velocity` in *opposite*
directions in the two modes (worse in `derived` at high sigma, better in
`ngs`) -- that divergence is the evidence the fix was correct, not just
a different wrong one.

**Open item, not yet run** *(resolved below — see the hybrid work and
the coverage-fix re-verification)*: Finding 2 suggests a hybrid
predictor — `toward_ball_land`'s geometric direction (robust to position
noise) combined with a Kalman-smoothed speed magnitude (robust the way
`kalman` is) — might dominate at every sigma tested, beating both the
destination-only and filter-only approaches across the whole range.

### Tier 1 Part 1 complete: hybrid, second noise axis, coverage fix, full-scale re-verification (2026-09-27/28)

Everything above this point in Tier 1 Part 1 was measured on week 1
only (128 plays) and, as of the coverage fix below, at a 3s horizon that
turned out to have 4.6% track coverage. Both limitations are now
resolved. Full write-up in README.md; summarised here.

**The hybrid predictor (`hybrid.py`) confirmed the open item above, with
one correction: not unconditionally.** Built the full 2×2 (raw/Kalman
anchor position × raw/Kalman speed) rather than just the hybrid cell, so
a win could be attributed to the right half. `tbl_kalman_speed` (speed
fixed, anchor untouched) beats every baseline tested past σ≈0.02–0.05
(week-1 3s figure) — `tbl_kalman_pos` (anchor fixed, speed untouched)
does nothing, confirming the win is the speed switch, not the anchor
fix. Mean signed along-track error explained the mechanism: raw-speed
destination methods flip from undershoot to a growing overshoot as σ
grows (their speed inflates fast enough to saturate the "never overshoot
the landing point" cap); Kalman-speed variants keep a bounded undershoot
instead. Fifth stated prediction in this project, and unlike four of the
previous five, it landed close to true.

**Second degradation axis (`degrade.add_ball_land_noise`,
`run_tier1_twoaxis.py`): the hybrid's immunity is to *player* noise
specifically, and that distinction matters.** Landing-point noise
(`sigma_ball`) drawn once per play from an independent RNG stream (not
once per frame — averaging across frames would hide a play-level bias).
Verified `sigma_ball=0` collapses the two-axis code path onto the
single-axis one exactly, and that non-destination predictors
(`constant_velocity`, `kalman`, `kalman_velocity`, `imm`, `kalman_ca`)
are provably invariant to `sigma_ball` — proven both ways, fires on a
deliberately broken path, passes on the correct one. `sigma_player`
dominates per-unit sensitivity by 6–43× depending on operating point
(checked the low end specifically, per instruction — it moved further
from parity with more data, not closer), but the ratio shrinks as noise
grows, because `sigma_player`'s slope saturates while `sigma_ball`'s
keeps rising. The engineering conclusion is weaker than the motivating
worry: real-world `sigma_ball` would need to be many times
`sigma_player` in absolute terms for "stop spending on localisation" to
follow, and this project has no measurement of that.

**The coverage-guard failure (see the new "Things that were considered
and rejected" entry above) forced re-verification of every claim in Tier
1 Part 1 at full scale.** `run_tier1_taskn.py` audited coverage across
all 18 weeks and chose 1.9s as the primary horizon (largest step at
≥50% coverage, 53.0%); 3s stays secondary, always reported at 3.5%
coverage now. Re-running every finding at both horizons on 2,729 plays
(not 128) surfaced one substantive reversal, not just tighter numbers:
**at 1.9s, clean data, the hybrid family loses to `constant_velocity`
(4.77 vs 4.57), and the old 3s-only "beats every baseline" headline was
true only past σ≈0.038, not at true zero noise.** (This entry originally
compared σ≈0.038 to "NGS's real accuracy range" — that range was later
found unsourced and the comparison is removed; see PART1_DETAIL.md's
2026-10-31 methodology note.) Along the coupled degradation scenario
(`σ_ball = σ_player`) the clean-data ranking fully inverts:
`velocity_turning_to_ball` (1st at σ=0) becomes worst at σ=5;
`tbl_kalman_speed` (4th at σ=0) becomes best. This is now the project's
strongest single result and leads the README's Tier 1 section.

**A related, smaller discrepancy, resolved rather than left implicit**:
`run_tier1_sweep.set_filter_R`'s `σ_pos = max(σ, 0.05)` floor (added in
Part 2 to keep `R` invertible at σ=0) means every degradation-sweep
table's "σ=0" row for a Kalman-family predictor uses `R_POS = 0.05²`,
not Tier 0b's UWB-sourced `0.25²` — a third, previously undocumented
reason a sweep's σ=0 doesn't equal the standalone Tier 0b number,
alongside the already-known finite-differencing floor. Verified by
forcing each `R` and reproducing both figures exactly. Not corrected —
the floored `R` is right for comparison within its own table, the
unfloored one is right as the Tier 0b figure, and a table shouldn't mix
them.

**`max(σ, 0.05)` vs. a physically-motivated alternative, checked
2026-09-28, `quadrature` then adopted as primary
(`run_tier1_r_convention_check.py`).** Flagged concern: in `derived`
mode, injected σ lands on positions that this project assumed (not yet
measured, at the time) carried NGS's own real error (~0.25 yd), so σ=0
shouldn't mean the filter trusts positions to 0.05 yd. Tested `σ_pos = sqrt(KALMAN_R_POS + σ²)` ("quadrature") against
the original `max(σ, 0.05)` ("floor") directly, full 18-week data. The
crossings moved materially — `kalman` 0.105→0.147 (+40%),
`tbl_kalman_speed` 0.028→0.075 (+162%). **Adopted `quadrature` as the
primary convention** (README.md states the reasoning in order: it's
physically correct — telling the filter `R=0.05²` at σ=0 states
something false about data that still carries NGS's own error; it
reproduces the standalone Tier 0b clean number exactly, which a
degradation sweep's zero point should; and it's the less flattering
choice, evidence it wasn't picked for the result). `floor` kept
throughout as a labelled sensitivity convention, not deleted — the
detailed σ-by-σ tables in Findings 2-4 stay on `floor` rather than being
re-run, justified by the two conventions converging above σ≈0.3
(numerically verified at σ=5: 0.1% difference). Consequence for the
headline: the project's lead crossing number is now `tbl_kalman_speed`
vs. `constant_velocity` (0.075) rather than plain `kalman`'s (0.147).
(This entry originally expressed both relative to "NGS's floor"/"NGS's
0.15-0.3 range"; that figure was later found unsourced and the
comparisons are removed — the crossings themselves, 0.075 and 0.147,
are unaffected.) Side effect, not the reason for the decision:
quadrature's σ=0 `R_POS` is `KALMAN_R_POS` exactly by construction, so it
resolves the earlier 5.25-vs-4.77 discrepancy automatically.

Also recorded in README.md, independent of either convention:
`output_*.csv` is itself NGS-measured, not physical ground truth — every
number in this project has always meant "distance from NGS's
measurement," not from where the player physically was.

**Next**: Task P (5-fold grouped cross-validation over the 231
train+val games, test split touched once at the end) before Task K (the
learned model). Both pending review.

### Known soft spots in the current code

- Acceleration direction is assumed along-track (`a` is a magnitude in the
  source data). Wrong during cuts, by construction. Documented in
  `data_loader.add_kinematics`.
- `_pivot_truth` (in `data_loader.py`) drops any play where a
  player-to-predict is missing a frame in the output window, rather than
  interpolating. Fine at Tier 0a; revisit if the drop count is large.
- `velocity_turning_to_ball` (in `baselines.py`) has an unfitted time
  constant (`tau = 0.8`). Deliberately not tuned — it's a baseline, not a
  model — but it's also the current @3s leader, so tuning it later would
  need a note that the comparison stops being baseline-vs-baseline.
- `KALMAN_R_VEL` (`config.py`) is a placeholder (σ = 0.5 yd/s) — no
  published spec for NGS's own velocity-channel noise, unlike `R_POS`
  which is UWB-sourced. `kalman_velocity`'s NIS (2.87 vs. an expected 4,
  even after `Q` was NIS-calibrated) suggests it's probably still
  oversized; Tier 1's R-sweep is the honest place to examine it, not
  here.
- IMM's manoeuvring model essentially never wins at the values actually
  used (`sigma_steady = 1.791`, ratio 10×) — see the Tier 0b Experiment
  A/B notes above. The 2-model IMM as built isn't doing useful regime
  detection on this dataset at these settings. A future pass could
  revisit the transition matrix or the ratio, but per `TIER0B_SPEC.md`'s
  "don't tune against results" and the round's own timebox, that's left
  alone here on purpose.
- The Kalman/IMM `sigma_a` and NIS numbers above were computed on week 1
  only. They're estimated from data at runtime, not hardcoded, so
  widening `WEEKS` recomputes them automatically on the next run — but
  the specific values printed in this file would go stale at that point.

## Working style notes

Roy prefers being told directly when something will not work, including
when the person he is talking to was the one who suggested it. Two
corrections already made in this project: an earlier assistant's advice was
built on a closed competition and named the wrong university, and a
silently-degrading bug in the acceleration baseline was caught only by
running the code rather than reading it. Both are the reason this file
exists.

Explanations land better with an analogy alongside the formal version, and
conclusions up front rather than a derivation that arrives at one.

Prefers decisions settled by a diagnostic run over a plausible-sounding
argument — the Tier 0b round is the clearest example: the NIS check and
the `kalman_ca` ablation against IMM were both requested specifically to
turn a reasonable hypothesis into a verified one, and both came back
counterintuitive (NIS-consistency made accuracy worse; IMM's gain turned
out to be model order, not switching). Timeboxes are meant literally —
report where things stand at the deadline rather than keep iterating, and
a null or negative result is treated as a complete, useful answer, not a
reason to extend scope.
