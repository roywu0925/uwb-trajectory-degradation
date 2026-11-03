# Tier 2A — drift detection, results so far

Working notes, not yet folded into README.md. Will be merged into one
coherent Tier 2 section once item 3 (sigma_hat estimator) is complete.

## Corrected bottom line (supersedes the first-pass report)

**Signal 1 (input PSI) is a working monitor for K5**, not a failure:

- alarm sigma 0.091, K5's hard failure (crosses the best of
  `constant_velocity`/`tbl_kalman_speed`/`velocity_turning_to_ball`) at
  0.921 — **lead +0.830 yd**.
- quiet on the verified-benign shift (0.0272 vs. threshold 0.2).
- threshold is the standard PSI convention (0.2), never tuned.

That is a pass on both metrics specified (sensitivity and specificity). The
original report's "none of the four signals is deployment-ready" was wrong
as written for K5 specifically — it holds for K3, not K5.

**One real caveat, not fatal**: K5's *soft* lead is thin (+0.069 yd) — signal
1 fires only slightly before K5's error starts growing past its own
clean-data baseline. It fires with much more room (+0.830) before K5's error
crosses the best physics baseline, which is the operationally relevant
threshold (the point where you'd actually want to switch predictors, not the
point where K5 first gets slightly worse than itself). The decision that
matters — replace K5 or not — is well covered.

## K3 reframed — not four monitoring failures, one model problem

K3's soft failure sigma is 0.00276 — below the published data's own
quantization step (0.01 yd, measured directly; see PART1_DETAIL.md's
methodology note). The data itself can't resolve a position change smaller
than 0.01 yd, and K3 has already failed by a factor of ~3.6x below that
resolution. No signal built on statistical detection can fire before a
real distributional shift exists to detect; at sigma=0.003 nothing has
moved enough to be measurable by any of the four constructions. This isn't
a monitoring gap, it's a model problem.

Confirmed by an independent calculation, not just restated: K3's velocity
gain (8.78) times the differencing amplification (sqrt(2)/dt ~= 14.14)
times sigma=0.003 gives an expected output perturbation of

    8.78 x 14.14 x 0.003 = 0.3725 yd

against a 10%-of-clean-error budget of 1.80 x 0.10 = 0.18 yd. 0.3725 > 0.18
— the arithmetic independently reproduces the empirically-measured failure
point (0.00276) from the gain and amplification numbers established
earlier in Tier 2, without reference to the monitoring signals at all.

**The conclusion for K3: monitoring cannot cover for a model this fragile.
The model has to be fixed (K4's cleaner input, or K5's noise-augmented
training) before deployment, not instrumented around.**
