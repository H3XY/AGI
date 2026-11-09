# Confirmatory evaluation (pre-registered: see PROTOCOL.md)

## sentiment: CONFIRMATORY set (validation), 1066 items, 50.0% class 1 (majority baseline 0.500)

| Model | Seeds 0, 1, 2 | Mean ± std |
|---|---|---|
| Quantum (QNN) | 0.769, 0.764, 0.765 | 0.766 ± 0.003 |
| Logistic regression, CV-tuned | 0.769, 0.775, 0.763 | 0.769 ± 0.006 |
| SVM (RBF), CV-tuned | 0.769, 0.766, 0.765 | 0.767 ± 0.002 |
| Full embedding, logistic regression, CV-tuned | 0.777, 0.777, 0.777 | 0.777 ± 0.000 |

| QNN minus | Mean diff | Two-level 95% CI (seeds, items) | Per seed: diff [paired CI], McNemar p |
|---|---|---|---|
| Logistic regression, CV-tuned | -0.003 | [-0.017, +0.011] | +0.000 [-0.014, +0.015], p=1; -0.011 [-0.025, +0.002], p=0.134; +0.003 [-0.014, +0.020], p=0.826 |
| SVM (RBF), CV-tuned | -0.001 | [-0.013, +0.012] | +0.000 [-0.015, +0.015], p=1; -0.003 [-0.018, +0.012], p=0.81; +0.001 [-0.017, +0.019], p=1 |
| Full embedding, logistic regression, CV-tuned | -0.011 | [-0.026, +0.004] | -0.008 [-0.024, +0.009], p=0.44; -0.013 [-0.031, +0.004], p=0.161; -0.011 [-0.027, +0.005], p=0.213 |

**Pre-registered verdict:** not confirmed (CI includes 0)

## sentiment: exploratory test set (as in the paper), 1066 items, 50.0% class 1 (majority baseline 0.500)

| Model | Seeds 0, 1, 2 | Mean ± std |
|---|---|---|
| Quantum (QNN) | 0.758, 0.765, 0.748 | 0.757 ± 0.009 |
| Logistic regression, CV-tuned | 0.774, 0.772, 0.770 | 0.772 ± 0.002 |
| SVM (RBF), CV-tuned | 0.766, 0.767, 0.775 | 0.770 ± 0.005 |
| Full embedding, logistic regression, CV-tuned | 0.774, 0.774, 0.774 | 0.774 ± 0.000 |

| QNN minus | Mean diff | Two-level 95% CI (seeds, items) | Per seed: diff [paired CI], McNemar p |
|---|---|---|---|
| Logistic regression, CV-tuned | -0.015 | [-0.029, -0.003] | -0.016 [-0.030, -0.002], p=0.0331; -0.008 [-0.021, +0.005], p=0.332; -0.023 [-0.038, -0.008], p=0.00631 |
| SVM (RBF), CV-tuned | -0.013 | [-0.030, +0.003] | -0.008 [-0.023, +0.006], p=0.306; -0.003 [-0.016, +0.010], p=0.784; -0.027 [-0.045, -0.012], p=0.00193 |
| Full embedding, logistic regression, CV-tuned | -0.017 | [-0.033, -0.001] | -0.016 [-0.033, +0.000], p=0.0712; -0.009 [-0.024, +0.007], p=0.295; -0.026 [-0.042, -0.011], p=0.00152 |

## cve: CONFIRMATORY set (holdout), 13432 items, 47.9% class 1 (majority baseline 0.521)

| Model | Seeds 0, 1, 2 | Mean ± std |
|---|---|---|
| Quantum (QNN) | 0.707, 0.700, 0.718 | 0.708 ± 0.009 |
| Logistic regression, CV-tuned | 0.720, 0.723, 0.716 | 0.720 ± 0.004 |
| SVM (RBF), CV-tuned | 0.729, 0.730, 0.720 | 0.726 ± 0.005 |
| Full embedding, logistic regression, CV-tuned | 0.725, 0.725, 0.725 | 0.725 ± 0.000 |

| QNN minus | Mean diff | Two-level 95% CI (seeds, items) | Per seed: diff [paired CI], McNemar p |
|---|---|---|---|
| Logistic regression, CV-tuned | -0.011 | [-0.024, +0.003] | -0.013 [-0.019, -0.006], p=7.97e-05; -0.023 [-0.029, -0.017], p=2.34e-13; +0.003 [-0.003, +0.009], p=0.361 |
| SVM (RBF), CV-tuned | -0.018 | [-0.031, -0.003] | -0.022 [-0.027, -0.017], p=9e-18; -0.030 [-0.037, -0.023], p=5.46e-20; -0.002 [-0.007, +0.004], p=0.536 |
| Full embedding, logistic regression, CV-tuned | -0.017 | [-0.027, -0.006] | -0.018 [-0.025, -0.010], p=8.12e-07; -0.025 [-0.032, -0.018], p=5.76e-12; -0.006 [-0.013, +0.000], p=0.0582 |

**Pre-registered verdict:** tie confirmed (CI within +/-3 points)

## cve: exploratory test set (as in the paper), 2424 items, 42.2% class 1 (majority baseline 0.578)

| Model | Seeds 0, 1, 2 | Mean ± std |
|---|---|---|
| Quantum (QNN) | 0.699, 0.680, 0.717 | 0.699 ± 0.019 |
| Logistic regression, CV-tuned | 0.692, 0.691, 0.683 | 0.689 ± 0.005 |
| SVM (RBF), CV-tuned | 0.702, 0.683, 0.689 | 0.691 ± 0.010 |
| Full embedding, logistic regression, CV-tuned | 0.702, 0.702, 0.702 | 0.702 ± 0.000 |

| QNN minus | Mean diff | Two-level 95% CI (seeds, items) | Per seed: diff [paired CI], McNemar p |
|---|---|---|---|
| Logistic regression, CV-tuned | +0.010 | [-0.013, +0.035] | +0.007 [-0.009, +0.023], p=0.427; -0.011 [-0.024, +0.002], p=0.121; +0.034 [+0.019, +0.049], p=7.94e-06 |
| SVM (RBF), CV-tuned | +0.007 | [-0.012, +0.028] | -0.003 [-0.017, +0.010], p=0.675; -0.003 [-0.017, +0.010], p=0.691; +0.028 [+0.015, +0.043], p=0.000103 |
| Full embedding, logistic regression, CV-tuned | -0.004 | [-0.026, +0.019] | -0.003 [-0.021, +0.014], p=0.748; -0.022 [-0.039, -0.006], p=0.0098; +0.015 [-0.002, +0.031], p=0.0898 |

