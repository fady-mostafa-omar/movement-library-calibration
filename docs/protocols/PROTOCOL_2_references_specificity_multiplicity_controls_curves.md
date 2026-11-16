# Pre-specification: analyses N1–N4 requested by the R.1 review

## Amendment 1 (12 Nov 2026, before any run): the implementation in `n1_n3.py` is adopted for N1–N3
**Status:** saved before the N1–N3 run; approved by Amr on 12 Nov 2026.

**Why:** a second session wrote a complete script for N1–N3 (`n1_n3.py`). It covers the same questions. Two of its choices are stronger than this draft:
- the low-frequency candidates use true causal 0.3–3 Hz amplitudes from the raw recording, on 61 channels as in Ofner et al. (2017), and on the 31 sites;
- the arm-trial features (N2) are computed from the raw recording, with a check that the same code reproduces the cached hand and rest features.

To keep one record, that script replaces the N1–N3 settings below wherever they differ. N4 and the A.11 check stay as written.

## Amendment 2 (12 Nov 2026, after N1–N3, before N4/N5): N5 added; N4 details fixed
**Status:** approved by Amr on 12 Nov 2026 (N5 and the new paper message), before N4 and N5 were run.

**Why:** N1 showed that the strong own-calibration pipeline is P1 (TS+LDA, 8 sites, windows from 1.0 s). The A.8 calibration curve used the weaker pipeline (P0), so its practical message needs the strong pipeline.

**N5 settings** (script `n4_n5.py`):
- **Data:** the A.8 design unchanged. Budgets are the first N T1 trials of MI runs 1–5, N = 10, 20, 30, 45, 60, all.
  - Own labelled uses the R4M-clean trials among them. A person is left out at a budget where a class has fewer than 2 trials (A.8 rule).
  - The hybrid prior weight is 20.
- **Methods**, each computed with windows from 0 s (must reproduce A.8) and from 1.0 s (the strong pipeline):
  - own labelled;
  - others' ME strict;
  - others' ME + recentering;
  - hybrid.
- **Anchors:**
  - 0–3 s reproduces A.8 (own 56.5…67.1; recentering 63.9…65.3; hybrid 65.5…66.0; strict 61.0);
  - from 1.0 s, own all = 75.8 (N1 P1) and recentered 60 = 68.6 (A.13a, which chose 1.0 s in all 15 folds).
- **Outcome:** for each mode, the smallest labelled budget at which the strong own calibration (from 1.0 s) matches or exceeds that mode's group mean. Paired differences with 95% CIs are given at every budget (descriptive). Modes:
  - TS+LDA zero calibration 0–3 s and from 1.0 s;
  - recentered 60 and hybrid 60 from 1.0 s;
  - ShallowConvNet strict, aligned 10, aligned 60 (A.10b).
- **Wording:** "a new user starts at the level that a well-tuned own calibration reaches after about N labelled trials (group means)". If the level is not reached within runs 1–5, say so.

**N4 settings:**
- ShallowConvNet, cell 4, strict, seeds 0–1, the A.4b-1/A.10b code unchanged. Inputs: frontal 5 (F3, F1, Fz, F2, F4) from the deep cache, and EOG 3.
- **EOG epochs** are built from the raw recording with the filter variant (causal Butterworth order 4, 0.3–40 Hz; with or without steady-state initial conditions) that reproduces the cached EEG epochs:
  - largest relative difference below 10⁻⁴;
  - the same trials, order and labels.

  If no variant reproduces them, the EOG control is not run and this is reported.
- CAR over the 3 EOG channels, as for every montage.
- **Pass rule:** mean below 55% and not significantly above 50% (one-sided tie-robust Wilcoxon, p ≥ 0.05).

**Settings now in force for N1–N3 (Amendment 1):**
- **N1 candidates:**

| # | Decoder |
|---|---|
| P0 | TS+LDA, 8 sites, 0–3 s (current reference) |
| P1 | TS+LDA, 8 sites, windows from 1.0 s |
| P2 | TS+LDA, 31 sites, 0–3 s |
| P3 | TS+LDA, 31 sites, windows from 1.0 s |
| P4 | Low-frequency amplitudes, 61 channels: causal 0.3–3 Hz, CAR, 15 samples from 0.2 to 3.0 s |
| P5 | As P4, on 31 sites |
| P6 | FBCSP + LDA, 31 sites, 0–3 s: 7 bands from 4 to 32 Hz, 2 CSP pairs per band, 10% shrinkage |
| P7 | As P6, 1–3 s |

- **Nested choice:** per person, by leave-one-run-out in runs 1–5 (unchanged).
- **Upper bound:** the best candidate chosen on the test runs, described as optimistic.
- **Primary-reference rule:** if the nested reference minus P0 is 2 points or more (group mean), the nested reference becomes the primary reference of the paper. This replaces the significance-based rule. It is the stricter test of our claim: the reference switches more easily.
- **Equivalence families (TOST ±5, Holm within each):**
  - A: TS+LDA, as in A.9;
  - B: ShallowConvNet, as in A.10;
  - C: the 10-20 site sets;
  - D: all modes vs the nested reference;
  - E: all modes vs the upper bound.
- **N2 verdict, per mode** (paired tie-robust Wilcoxon, two-sided):

| Arm vs hand | Arm detection rate | Verdict |
|---|---|---|
| Not lower (p ≥ 0.05) | — | **Not hand-specific.** The title and claims change to "upper-limb movement imagery" |
| Lower | Below the midpoint between hand and rest | **Hand-specific** |
| Lower | At or above that midpoint | **Partly specific** |

  The strict mode decides the wording of the main claim.
- **Safety:**
  - every earlier result is reproduced (anchors);
  - the trials and features computed from the raw recording must match the cache (same trials and order; largest feature difference ≤ 10⁻²);
  - otherwise the run stops.

**Three corrections made to the script (12 Nov 2026, before running):**
1. The Wilcoxon normal approximation is called `asymptotic` in current scipy; `approx` would stop the summary with an error.
2. The ShallowConvNet per-person values are read from `A10b_results.csv`, not typed rounded values.
3. The A.11 K = 12 check was added at the end.

**Dropped from the earlier draft:**
- the 0.5-s window start;
- the low-frequency and FBCSP candidates on 8 sites;
- the 32–40 Hz band;
- Holm across the N2 tests.
**Written:** 12 Nov 2026, before any of these analyses was run · **Status:** approved by Amr, 12 Nov 2026 (before any run) · **Source:** `R1_REVIEW_RESULT.md` (C1, M1, M5, C1b) · **Outputs (planned):** `classification_MI-ME/30_v06_pipeline/16_N1_N4/`

## Why
The R.1 review found that two claims are weaker than written:
1. **"Equivalent to full calibration" (C1).** The reference (67.1%) is one labelled pipeline, TS+LDA on the 8 sites chosen for transfer. A stronger within-person decoder may exist on the same data.
2. **"Hand movement intention" (M1).** It is not shown that the detector responds to hand imagery rather than to any arm imagery.

It also asked for multiplicity control over the equivalence tests (C1b, M5) and for deep-model eye and frontal controls (M1).

All 15 people's results are known. So:
- every choice below is fixed **now**;
- no setting is chosen on runs 6–10 of the person being tested;
- every result is reported, whatever it shows;
- nothing outside this document is run.

## Common setup (unchanged)
- **Data:** Ofner T1 (hand open + close vs rest), 15 people, leave-one-subject-out.
- **Test set:** the new person's MI runs 6–10, all trials.
- **Own calibration** uses the person's labelled, artifact-free (R4M-clean) trials of MI runs 1–5, as in A.9.
- **Statistics:**
  - paired tests with the tie-robust Wilcoxon of the final run (differences rounded to 10⁻⁶; exact without ties or zeros, otherwise normal approximation with tie correction);
  - TOST with a ±5-point margin and 90% CI, as in A.9. The margin equals the smallest effect of interest fixed in plan rule 6 (+5 points, 7 Nov 2026);
  - Holm within each family named below.
- **Anchors** (must reproduce, otherwise the run stops):
  - TS+LDA own calibration, fold-specific 8 sites, 0–3 s: 67.1;
  - strict zero calibration: 61.0;
  - recentered 60: 65.4.

## N1: stronger within-person references (CPU)
**Question:** how good is the best labelled within-person decoder on this split, and how far are the zero-calibration and adapted modes from it?

**Candidate decoders** (train on labelled MI runs 1–5, test on runs 6–10, T1). All settings fixed now:

| # | Decoder | Sites | Features |
|---|---|---|---|
| 1 | TS+LDA (current reference) | fold-specific 8 | as A.9, windows starting 0.0–2.0 s |
| 2 | TS+LDA | fold-specific 8 | windows starting at or after 0.5 s |
| 3 | TS+LDA | fold-specific 8 | windows starting at or after 1.0 s |
| 4 | TS+LDA | all 31 pool sites | as A.9 |
| 5 | Low-frequency amplitude + shrinkage LDA | fold-specific 8 | CAR within the montage; mean amplitude in 12 bins of 0.25 s from 0 to 3 s after the cue (slow-potential features, in the spirit of Ofner et al. 2017) |
| 6 | Low-frequency amplitude + shrinkage LDA | all 31 | as 5 |
| 7 | FBCSP + shrinkage LDA | fold-specific 8 | causal Butterworth (order 4) bands 4–8, 8–12, 12–16, 16–20, 20–24, 24–28, 28–32, 32–40 Hz; CSP with 2 filter pairs per band fitted on the calibration trials; log-variance (32 features), no feature selection |
| 8 | FBCSP + shrinkage LDA | all 31 | as 7 |

- Decoders 5–8 use the A.4b-1 time-series cache: causal 0.3–40 Hz, cue to +3 s, 128 Hz. No other filters, windows or parameters are tried.
- Shrinkage LDA: `lsqr`, automatic shrinkage, equal priors, as everywhere else.

**Outcomes:**
1. **Primary reference, chosen without the test runs:** per person, the candidate with the highest leave-one-run-out accuracy within MI runs 1–5 (train on 4 runs, test on the 5th; ties go to the lower number). It is then trained on all of runs 1–5 and scored on runs 6–10. This is the **best within-person reference**.
2. **Secondary:** each candidate's test accuracy. Also "best on test", the per-person maximum over the 8 candidates, described as an optimistic upper bound because it is chosen on the test runs.
3. **Gain of the best reference over the matched reference** (best − candidate 1): paired Wilcoxon, two-sided.
4. **Distance of each mode from the best reference.** Modes:
   - TS+LDA strict, recentered 60, hybrid 60, and recentered 60 with the A.13a window;
   - ShallowConvNet strict, aligned 10, aligned 60 (A.10b, mean of 5 seeds).

   For each: mean difference, 90% CI and TOST (±5). Holm over these 7 TOSTs (family "N1").

**Pre-specified wording rule:**
- If the best reference is significantly higher than the matched reference (outcome 3: p < 0.05 and a positive difference), the paper's primary comparison becomes the best reference. "Equivalent to full calibration" is then used only where its Holm-adjusted TOST supports it. Otherwise it is replaced by "reaches X% of the best within-person decoder".
- If not, the matched reference stays primary and the best reference is reported next to it.

## N2: specificity of the detector (CPU)
**Question:** does the hand-vs-rest detector also respond to imagery of the other arm movements?

- **Trials:** the new person's MI runs 6–10 of the four arm classes (elbow flexion, elbow extension, supination, pronation; 6 per class per run, 120 trials), next to the hand trials (60) and rest trials (30).
- **Detectors:**
  - TS+LDA cell 4, strict and recentered 60, exactly as A.9. The recentering mean comes from the first 60 T1 trials of runs 1–5, as before.
  - The detectors are not retrained.
- **Features for the arm trials:** computed with the code that built the A.2b cache.
  - **Check (fixed now):** the same code, run on the person's hand and rest trials, must reproduce the cached features (largest relative difference below 10⁻⁶). Otherwise the run stops.
- **Outcome:** detection rate (% of trials called "movement") for hand trials (= hit rate), arm trials and rest trials (= false-alarm rate), per person and mean.
- **Tests per mode:**
  - arm vs hand;
  - arm vs rest.

  Paired tie-robust Wilcoxon, two-sided, Holm over the 4 tests (2 modes × 2).
- **Pre-specified verdict, per mode:**

| Arm vs hand | Arm vs rest | Verdict |
|---|---|---|
| Not lower (Holm p ≥ 0.05, or higher) | — | **Not hand-specific.** The title and claims use "upper-limb movement imagery vs rest" |
| Lower (Holm p < 0.05) | Higher (Holm p < 0.05) | **Hand-preferential** |
| Lower | Not different | **Hand-specific** |

  The strict mode decides the wording of the main claim.

## N3: multiplicity and equivalence (CPU, from saved results only)
1. **Holm within the existing TOST families**, recomputed from the result files:
   - A.9: the 7 TOSTs;
   - A.10: the 5 TOSTs.
2. **TOST (±5) for the 10-20 site sets** (Table 3: 6 differences), Holm over the 6.
3. **Wording rule:**
   - "Equivalent" is written only where the Holm-adjusted TOST p < 0.05.
   - Otherwise: "within X points (90% CI [a, b]); equivalence not shown after correction".
   - The A.9 primary test (recentered 60 vs own calibration) is reported first, as pre-specified then.

## N4: deep-model eye and frontal controls (GPU, short)
- **Model:** ShallowConvNet, cell 4, strict, the A.4b-1c code unchanged, seeds 0–1 (as the other deep controls).
- **Inputs:**
  1. frontal 5 sites only (F3, F1, Fz, F2, F4);
  2. the 3 EOG channels only, if they are in the deep time-series cache. If they are not, they are built with the cache code and the same check as N2.
- **Pass rule (fixed now):** a control passes if its mean balanced accuracy is below 55% and not significantly above 50% (one-sided tie-robust Wilcoxon, p ≥ 0.05). Each result is reported as is.
- **Secondary (only if N2's arm-trial features are available in time-series form):** ShallowConvNet strict detection rates on arm, hand and rest trials. Descriptive only.

## Checks requested by the review (no new analysis)
- A.11 at K = 12 vs A.10b:
  - recompute the ShallowConvNet K = 12 means from `A1113_deep_library.csv` (repeats 0–2);
  - set them next to seeds 0–2 of the repeat run and of A.10b, to confirm the values are not copied.
- The mixed deep-model runs in Table 2 (review m1) are resolved by wording: the source is stated per column.

## Runs
1. **Inspection (now, seconds):** print the keys and shapes of the data caches and the source of the scripts that built them (`a2b1.py`, `a4b1.py`, `a4b1c.py`). This only reads files. It is needed to write N1, N2 and N4 with the exact cache code.
2. **CPU run (one script):** anchors, N1, N2, N3, the A.11 check.
3. **GPU run (one script, about 10 min):** N4.

## What is not done
- No other decoders, bands, bins, windows, site sets, margins or thresholds.
- No retuning of the zero-calibration pipelines.
- The pre-specified primary pipeline of Paper 2 does not change.
- If a result weakens a claim, the claim is reworded as the rules above say.

## Approval
- [x] Approved by Amr on 12 Nov 2026, before any of these analyses was run.
