# Hardware repeat study: cve on ibm_marrakesh

40 test samples, 1024 shots, 3 repeats, same trained weights.

| | Value |
|---|---|
| Ideal simulator accuracy | 0.725 |
| Hardware accuracy per run | 0.725, 0.675, 0.700 |
| Answers flipped vs ideal, per run | 2, 2, 1 |
| Flips expected per run from shot noise alone | 1.15 |
| Samples flipped in every run | 0 |
| Samples flipped at least once | 3 |
| Mean shift of P(class 1), hardware minus ideal | -0.0060 |
| Mean shrink toward 0.5 | +0.0079 |

## Samples that flipped

| Sample | Label | P ideal | P hardware (per run) | Shot-noise flip prob. | Flip rate |
|---|---|---|---|---|---|
| 13 | 0 | 0.480 | 0.509, 0.501, 0.475 | 0.100 | 0.67 |
| 24 | 1 | 0.504 | 0.503, 0.498, 0.477 | 0.399 | 0.67 |
| 32 | 1 | 0.478 | 0.514, 0.497, 0.477 | 0.084 | 0.33 |
