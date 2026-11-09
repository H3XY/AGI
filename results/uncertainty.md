# Uncertainty and robustness checks

## sentiment (1066 test texts, seeds [0, 1, 2])

| Model (same 4 features unless noted) | Mean ± std over seeds | 95% bootstrap CI (seed 0) |
|---|---|---|
| Quantum (QNN) | 0.757 ± 0.007 | [0.732, 0.783] |
| Logistic regression, defaults | 0.770 ± 0.000 | [0.743, 0.795] |
| Logistic regression, CV-tuned | 0.772 ± 0.002 | [0.748, 0.799] |
| SVM (RBF), CV-tuned | 0.770 ± 0.004 | [0.740, 0.792] |
| Logistic regression, full embedding, defaults | 0.774 | |
| Logistic regression, full embedding, CV-tuned (C=1) | 0.774 | |

Paired 95% CI, QNN minus logistic regression (seed 0): defaults [-0.027, +0.006], CV-tuned [-0.030, -0.002].

## cve (2424 test texts, seeds [0, 1, 2])

| Model (same 4 features unless noted) | Mean ± std over seeds | 95% bootstrap CI (seed 0) |
|---|---|---|
| Quantum (QNN) | 0.699 ± 0.015 | [0.679, 0.717] |
| Logistic regression, defaults | 0.688 ± 0.004 | [0.673, 0.710] |
| Logistic regression, CV-tuned | 0.689 ± 0.004 | [0.673, 0.710] |
| SVM (RBF), CV-tuned | 0.691 ± 0.008 | [0.684, 0.719] |
| Logistic regression, full embedding, defaults | 0.715 | |
| Logistic regression, full embedding, CV-tuned (C=100) | 0.702 | |

Paired 95% CI, QNN minus logistic regression (seed 0): defaults [-0.009, +0.023], CV-tuned [-0.009, +0.023].

## Hardware repeats: cve (ibm_marrakesh, 40 inputs, 1024 shots, 3 runs)

- Expected flips per run from shot noise alone (exact binomial): 1.15
- Observed flips: 5; P(at least that many | shot noise only) = 0.244
- Accuracy per run: 0.725 [0.572, 0.839], 0.675 [0.520, 0.799], 0.700 [0.546, 0.819] (Wilson 95% CI)
- Mean signed shift in P(class 1), hardware minus ideal: -0.0060, 95% CI [-0.0104, -0.0014]
- Including the first benchmark run (4 runs): 8 flips vs 4.59 expected, P = 0.063

## Hardware repeats: cve_fake (fake_manila, 40 inputs, 1024 shots, 3 runs)

- Expected flips per run from shot noise alone (exact binomial): 1.15
- Observed flips: 7; P(at least that many | shot noise only) = 0.035
- Accuracy per run: 0.700 [0.546, 0.819], 0.675 [0.520, 0.799], 0.675 [0.520, 0.799] (Wilson 95% CI)
- Mean signed shift in P(class 1), hardware minus ideal: -0.0298, 95% CI [-0.0403, -0.0196]

