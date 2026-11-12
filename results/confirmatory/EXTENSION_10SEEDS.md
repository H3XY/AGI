# 10-seed extension of the confirmatory evaluation (specified before running)

Written and committed before any of the added seeds was trained or evaluated.

## Why, and how it relates to the original protocol

`PROTOCOL.md` fixed training seeds 0, 1, 2. A reviewer noted that the seed component
of the two-level bootstrap interval is poorly estimated from three seed clusters. This
extension adds seeds 3-9 to estimate training-seed variation better. It is a
**deviation from the pre-registered protocol** and is reported as a sensitivity
analysis. The primary, pre-registered verdicts remain the 3-seed ones in
`confirmatory.md`, whatever this extension shows.

## What is fixed

- System, training procedure, baselines, tuning grids and data: exactly as in
  `PROTOCOL.md` and `benchmark/confirmatory.py` (the `--extend` flag only adds seeds
  3-9; seeds 0-2 reuse the saved predictions, which are deterministic).
- Confirmatory sets: the same Rotten Tomatoes validation split and NVD May-Aug 2024
  holdout already used for the primary evaluation.
- Analysis: the same crossed two-level bootstrap (10,000 replicates), now over 10 seed
  clusters, with the same per-seed paired CIs and McNemar tests.
- Decision rules, applied unchanged and reported whichever way they go:
  sentiment, the QNN-minus-tuned-LR interval entirely below 0; CVE, entirely within
  [-0.03, +0.03].
- The dequantization surrogate (`benchmark/dequantize.py`) is extended to the same 10
  seeds with its pre-registered reading rule unchanged.

No other analyses are added, and no seeds will be dropped.
