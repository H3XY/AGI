# Quantum vs classical benchmark

Test accuracy. All models see the same reduced features (one per qubit).

| Dataset | Qubits | Train / test | Quantum (ideal sim) | Quantum (noisy IBM sim) | Logistic regression | SVM (RBF) |
|---|---|---|---|---|---|---|
| breast_cancer | 4 | 300 / 171 | 0.918 | 0.912 | 0.947 | 0.942 |
| iris | 4 | 70 / 30 | 0.900 | 0.867 | 1.000 | 1.000 |
| moons | 2 | 210 / 90 | 0.844 | 0.844 | 0.800 | 0.956 |
| sentiment | 4 | 300 / 200 | 0.725 | 0.725 | 0.765 | 0.750 |
| cve | 4 | 300 / 200 | 0.685 | 0.690 | 0.695 | 0.700 |

Noisy IBM sim = the ideal-trained weights run on `FakeManilaV2` (a local simulation of a 5-qubit IBM device's noise).

## Real IBM hardware

Same trained weights, same test samples (a subset, to save quantum time).

| Dataset | Device | Samples | Quantum (ideal sim) | Quantum (IBM hardware) | Difference |
|---|---|---|---|---|---|
| breast_cancer | ibm_marrakesh | 40 | 0.875 | 0.875 | +0.000 |
| iris | ibm_marrakesh | 30 | 0.900 | 0.900 | +0.000 |
| moons | ibm_marrakesh | 40 | 0.900 | 0.900 | +0.000 |
| sentiment | ibm_marrakesh | 40 | 0.675 | 0.675 | +0.000 |
| cve | ibm_marrakesh | 40 | 0.725 | 0.650 | -0.075 |

With 40 samples, one sample is 0.025 of accuracy, so a difference of one or two samples can come from shot noise alone.
