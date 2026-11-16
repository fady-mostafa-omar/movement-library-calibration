# Pre-registration: A.11 (library size), A.12 (operating point and AUC), A.13 (two accuracy attempts)
**Written:** 12 Nov 2026, before any of these analyses was run · **Status:** approved by Amr, 12 Nov 2026 (before any run) · **Outputs (planned):** `classification_MI-ME/30_v06_pipeline/14_A11_13/`

## Why
These analyses address three weaknesses of Paper 2 found in A.9/A.10:
1. **Library size:** would more healthy people raise accuracy?
2. **False alarms:** 30–45% of rest trials trigger a detection at the default rule.
3. **Modest accuracy:** 61–68%.

All 15 people's results are already known. So:
- every choice below is fixed **now**;
- every model choice uses training people only (rule 1);
- every result is reported, whatever it shows;
- no variant outside this document is tried.

## Common setup (unchanged from A.8–A.10)
- **Data and test set:** Ofner T1 (hand open + close vs rest). Test = the new person's MI runs 6–10, all trials. Leave-one-subject-out over 15 people.
- **Library:** other people's ME (cell 4), R4M-clean trials.
- **TS+LDA:** exactly as A.9:
  - fold-specific 8 sites;
  - causal 0.3–3 and 8–30 Hz;
  - 21 one-second windows over cue + 0–3 s (starts 0.0–2.0 s, step 0.1 s), trial decision from all 21;
  - CAR by projection; log-Euclidean tangent features;
  - shrinkage LDA, equal priors.
- **TS+LDA modes:**
  - **strict;**
  - **recentered:** each training session centred on its own mean; the new person centred on the first 60 trials of MI runs 1–5, without labels.
  - **Own calibration** (cell 1): own MI runs 1–5, labelled.
- **ShallowConvNet:** exactly the A.10b code:
  - two early-stopping people chosen by `rng(100·t + seed)` and excluded from training;
  - **strict;**
  - **aligned:** Euclidean alignment; the new person aligned on the first 60 trials of MI runs 1–5 ("aligned 60").
- **Statistics:** paired Wilcoxon, two-sided. Holm correction only where stated. 95% CIs are t-based across the 15 people.

## A.11: library-size curve
**Question:** how does zero-calibration accuracy change with the number of people in the library?

**TS+LDA (CPU):**
- **Library sizes:** K = 2, 4, 6, 8, 10, 12 and 14 people, drawn from the new person's 14 others.
- **Subsets:** for K < 14, 10 random subsets per person (`numpy.random.default_rng(1000·t + K)`, without replacement). K = 14 is the full library.
- **Modes:** strict and recentered (60 trials).
- **Sites:** the fold-specific 8, selected earlier with all 14 others, are kept for every K.

**ShallowConvNet (GPU):**
- **Early stopping:** the two early-stopping people are chosen as in A.10b for repeat r = 0, 1, 2 (`rng(100·t + r)`).
- **Library sizes:** K = 3, 6, 9 and 12 training people, drawn from the remaining 12 (`rng(1000·t + 10·K + r)`). Seed = r.
- **Modes:** strict and aligned 60.
- K = 12 is identical to A.10b seeds 0–2 (anchor).

**Outcomes:**
- **Curve:** mean accuracy vs K, with 95% CI, per pipeline and mode. Each person's value is the mean over subsets or repeats.
- **Points per doubling:** slope b of accuracy = a + b·log2(K), fitted by least squares to the person-level values, with a bootstrap 95% CI over people (2,000 resamples). It is descriptive only; no extrapolation beyond the tested range is claimed.
- **Pre-specified verdict per curve:**
  - **"still rising"** if the gain from the second-largest to the largest K is ≥ 1.0 point **and** the CI of b excludes 0;
  - **"plateau"** if that gain is < 0.5 point;
  - otherwise **"inconclusive"**.

**Anchors:**
- TS+LDA K = 14 reproduces A.9 (strict 61.0; recentered 60 65.4).
- ShallowConvNet K = 12 agrees with A.10b seeds 0–2 within GPU noise (group means within 0.5).

## A.12: operating point and AUC
**Question:** how well do the detectors separate movement from rest, independent of the threshold? With a threshold chosen without the new person to keep false alarms at 20%, how many movements are still detected?

**Scores:**
- **TS+LDA:** LDA log-odds per trial.
- **ShallowConvNet:** softmax probability of movement per trial and seed.

**Configurations:**
- TS+LDA strict, recentered 60, own calibration;
- ShallowConvNet strict, aligned 60.

**AUC:**
- Per person, on the test trials, with movement as the positive class.
- ShallowConvNet: AUC per seed, then the mean over the 5 seeds.

**Deployable threshold** (target false-alarm rate α = 20% primary, 10% secondary), chosen without the new person:
- **TS+LDA strict and recentered:**
  - inner leave-one-subject-out over the 14 training people;
  - each is scored by a model trained on the other 13 people's ME, on his MI runs 6–10 (recentered on his own first 60 MI trials in the recentered mode);
  - the threshold is the (1 − α) quantile of the pooled rest-trial scores.
- **TS+LDA own calibration:** leave-one-run-out within MI runs 1–5 (train on 4 runs, score the 5th); same quantile rule.
- **ShallowConvNet:** the (1 − α) quantile of the rest-trial scores of the two early-stopping people (their MI runs 6–10, aligned as in training), per seed.
- **Note:** this uses other people's MI labels to set one number (the threshold), as A.2b used them for site selection and A.4b for early stopping. The new person's data are never used.

**Outcomes:**
- **Primary:** at the deployable 20% threshold, the mean achieved false-alarm rate and hit rate on the new people, and the number of people whose false-alarm rate exceeds 30%.
- **Secondary:**
  - the same at 10%;
  - AUC per configuration;
  - paired AUC comparisons, Holm over 2: ShallowConvNet strict vs TS+LDA strict, and ShallowConvNet aligned 60 vs TS+LDA recentered 60;
  - mean ROC curves for the figure.

**Anchor:** TS+LDA decisions at log-odds 0 reproduce A.9 balanced accuracy.

## A.13: two accuracy attempts (only these two)

### A.13a: analysis window chosen on training people (TS+LDA)
- **Rationale:** in A.9, the first second after the cue carried no information and dilutes the trial decision.
- **Candidates:** trial decision from the windows that start at or after s, for s = 0.0 (all 21 windows; current), 0.5 (16 windows) or 1.0 s (11 windows). All windows end by 3.0 s.
- **Choice:**
  - per outer fold, by inner leave-one-subject-out over the 14 training people (train on 13 people's ME, test on the 14th's MI runs 6–10);
  - mean balanced accuracy; ties go to the smallest s;
  - separately for strict and recentered 60.
- **Test:** nested-choice accuracy vs the current 0–3 s rule, per mode; paired Wilcoxon, Holm over 2 modes.
- **Claim rule:** "improves accuracy" only if Holm-adjusted p < 0.05 and the mean difference is positive.
- **Note:** the decision is still made at 3.0 s. No latency gain is claimed.

### A.13b: fixed ensemble of the two pipelines
- **Probabilities:**
  - TS+LDA: posterior P(movement) = 1 / (1 + exp(−log-odds));
  - ShallowConvNet: P(movement) averaged over the 5 seeds.
- **Ensemble:** P = (P_TS + P_deep) / 2; movement if P > 0.5. No weights or thresholds are tuned.
- **Configurations:**
  - strict (TS+LDA strict + ShallowConvNet strict);
  - adapted (TS+LDA recentered 60 + ShallowConvNet aligned 60).
- **Test:** ensemble vs ShallowConvNet alone (the better single model), per configuration; paired Wilcoxon, Holm over 2. Ensemble vs TS+LDA is also reported.
- **Claim rule:** "improves accuracy" only if Holm-adjusted p < 0.05 and the mean difference is positive.
- **Also reported:** hit, false alarm and AUC of the ensemble.

## Runs
1. **GPU (one script, about 25 min on L4):**
   - ShallowConvNet strict and aligned (10, 60, all), 5 seeds, saving per-trial probabilities for the test person and the two early-stopping people (A.12, A.13b);
   - the A.11 library curve (K = 3, 6, 9, 12; r = 0–2; strict and aligned 60).
   - Balanced accuracy of this repeat run is checked against A.10b (group means within 0.5). **Paper 2 keeps A.10b for deep balanced accuracy.**
2. **CPU (one script):**
   - A.11 TS+LDA, A.12 TS+LDA, A.13a;
   - then, from the GPU outputs: A.11 deep summary, A.12 deep, A.13b.

## What is not done
- No other window starts, ensemble weights, α levels, models, band settings or site sets.
- No hybrid in A.12/A.13.
- No PhysioNet data.
- If an attempt fails, it is reported as a failure. The pre-specified primary pipeline of Paper 2 (TS+LDA, fold-specific 8, 0–3 s) does not change. Successful A.13 attempts are reported as pre-registered secondary analyses.

## Approval
- [x] Approved by Amr on 12 Nov 2026, before any of these analyses was run.
