# Confirmatory evaluation protocol (pre-registered)

Written and committed **before** the confirmatory data was downloaded or any
confirmatory result was computed. The git commit that adds this file is the timestamp.

## Why

The circuit design and the feature reduction (PLS) were chosen in exploratory runs
that reported test-set accuracy. The test-set numbers in the paper are therefore
exploratory and may be optimistic. This evaluation freezes everything and scores it
once on data that played no part in any design decision.

## Frozen system (no changes allowed after this commit)

- Code: `benchmark/confirmatory.py` at this commit, which calls the same functions as
  `benchmark/uncertainty.py` (`make_angle_pipeline(4, seed=0, method="pls")` fitted on
  the full training split; `build_qnn(num_qubits=4, reps=2)`; `train_classifier(...,
  maxiter=60, seed=s)` on the balanced 600-text training subset of seed `s`).
- Training seeds: 0, 1, 2. Training data: the existing training splits only.
- Baselines, same 4 features: logistic regression and RBF SVM, tuned by 5-fold CV on
  the same 600 training texts with the grids in `benchmark/uncertainty.py`.
  Reference: logistic regression on the full 384-dim embedding, all training texts,
  C tuned by 5-fold CV.
- Prediction rule: class 1 iff P(class 1) > 0.5 (exact probabilities).

## Confirmatory data (never used before this commit)

- **sentiment:** the Rotten Tomatoes `validation` split (1,066 sentences), a split of
  the same dataset that this project never downloaded until now.
- **cve:** every CVE with a CVSS v3.1 score published 1 May - 28 Aug 2024 (after the
  Jan-Apr 2024 train/test window), from the NVD API, same labelling rule
  (score >= 7.0 = high/critical). The share of high/critical will be reported, since
  it may differ from training.

## Analysis (fixed in advance)

Primary comparison: QNN minus CV-tuned logistic regression on the same 4 features,
accuracy averaged over seeds 0-2.

- 95% CI by a two-level bootstrap: resample the 3 seeds with replacement, then the
  test items with replacement (the same items for every model), 10,000 resamples.
- Per seed: paired bootstrap 95% CI over items and an exact McNemar test.

Pre-stated expectations, from the exploratory results:

- **sentiment:** the QNN is below tuned logistic regression (exploratory difference
  about -1.5 points). Confirmed if the two-level 95% CI lies entirely below 0.
- **cve:** no meaningful difference. Equivalence margin +/-3 points: "tie" is
  confirmed if the two-level 95% CI lies entirely within [-0.03, +0.03]; a CI that
  excludes 0 on either side is reported as a difference; anything else is
  inconclusive.

All outcomes are reported, whichever way they go. Secondary: the same contrasts
against the tuned SVM and the full-embedding reference, and the exploratory test-set
numbers recomputed with the same per-seed paired analysis for comparison.

## Outputs

`results/confirmatory/confirmatory.json` and `confirmatory.md`; per-item predictions in
`results/confirmatory/predictions_<task>.npz`.
