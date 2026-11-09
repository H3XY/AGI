# Hardware repeat study: cve on fake_manila

40 test samples, 1024 shots, 3 repeats, same trained weights.

| | Value |
|---|---|
| Ideal simulator accuracy | 0.725 |
| Hardware accuracy per run | 0.700, 0.675, 0.675 |
| Answers flipped vs ideal, per run | 3, 2, 2 |
| Flips expected per run from shot noise alone | 1.15 |
| Samples flipped in every run | 2 |
| Samples flipped at least once | 3 |
| Mean shift of P(class 1), hardware minus ideal | -0.0298 |
| Mean shrink toward 0.5 | +0.0309 |

## Samples that flipped

| Sample | Label | P ideal | P hardware (per run) | Shot-noise flip prob. | Flip rate |
|---|---|---|---|---|---|
| 10 | 1 | 0.510 | 0.488, 0.475, 0.469 | 0.250 | 1.00 |
| 15 | 0 | 0.530 | 0.487, 0.510, 0.519 | 0.025 | 0.33 |
| 24 | 1 | 0.504 | 0.465, 0.477, 0.477 | 0.399 | 1.00 |
