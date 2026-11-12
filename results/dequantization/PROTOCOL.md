# Dequantization check: protocol (written before any result was computed)

## Question

Can a classical model that uses the quantum circuit's own function class, a
trigonometric polynomial in the inputs, match the QNN on the two text tasks? If it
can, the QNN offers nothing a classical surrogate cannot reproduce cheaply
(Schuld et al. 2021; Landman et al. 2022).

## Why the surrogate is exact in form

The QNN encodes each of its 4 input angles x_i with R_Y(x_i), twice (data
re-uploading). The generator of R_Y has eigenvalues +-1/2, so each encoding gate
contributes frequencies {-1, 0, 1} and two re-uploads give {-2, -1, 0, 1, 2} per
feature. Every output probability of the circuit, for any weights, is therefore

    f(x) = sum over w in {-2..2}^4 of  a_w cos(w . x) + b_w sin(w . x)

with 313 distinct frequency vectors up to sign (625 including negatives), so at most
625 real coefficients. The check `spectrum_check` in `benchmark/dequantize.py` verifies
this numerically: it fits that basis by least squares to the trained QNN's exact output
on random inputs and reports the residual (expected ~0).

## Classical surrogates (fixed in advance)

Same inputs as the QNN: the 4 PLS features scaled to [0, pi], the same balanced
600-text training subset per seed (seeds 0, 1, 2), same label rule.

1. **Fourier surrogate (full spectrum):** logistic regression on all 624 cos/sin
   features plus intercept, C tuned by 5-fold CV on the 600 training texts
   (C in {0.001, 0.01, 0.1, 1, 10, 100}).
2. **Random Fourier subsets:** the same, but on k frequency vectors drawn at random
   (seeded) from the 312 non-zero ones, k in {8, 32, 128}; this mirrors the
   random-Fourier-feature dequantization of Landman et al.

## Evaluation

On the exploratory test sets and on the confirmatory sets (Rotten Tomatoes validation,
NVD CVEs May-Aug 2024). The QNN's predictions are taken unchanged from
`results/confirmatory/predictions_<task>.npz` (same seeds and training subsets).
Primary contrast: QNN minus full Fourier surrogate on the confirmatory sets, two-level
bootstrap over seeds and items (10,000 resamples).

Reading, fixed in advance: if the surrogate's two-level 95% CI for (QNN - surrogate)
lies entirely at or below +0.01, the QNN is "classically matched" on that task (the
surrogate is no more than 1 point worse); if the CI lies above 0, the surrogate
underperforms the QNN. Results are reported whichever way they go.
