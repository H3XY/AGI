# Cross-device replication: CVE classifier, 100 identical inputs

Noise-free accuracy on these inputs: 0.650. Expected changed predictions from shot noise alone: 3.09 per run. Marrakesh rows are its three runs restricted to these 100 inputs.

| Backend | Qubits | Depth / 2q gates | Readout err | 2q err | Accuracy | McNemar vs noise-free (dev-only/ideal-only, p) | Changed (p) | Mean shift [95% CI] |
|---|---|---|---|---|---|---|---|---|
| ibm_marrakesh run 1 | [0, 1, 2, 3] | 37 / 6 | 0.0059-0.0239 | 0.0022-0.0027 | 0.650 | 2/2, p=1 | 4 (p=0.373) | -0.0066 [-0.0105, -0.0025] |
| ibm_marrakesh run 2 | [0, 1, 2, 3] | 37 / 6 | 0.0059-0.0239 | 0.0022-0.0027 | 0.600 | 1/6, p=0.125 | 7 (p=0.0185) | -0.0066 [-0.0107, -0.0026] |
| ibm_marrakesh run 3 | [0, 1, 2, 3] | 37 / 6 | 0.0059-0.0239 | 0.0022-0.0027 | 0.660 | 3/2, p=1 | 5 (p=0.174) | -0.0019 [-0.0054, +0.0016] |
| ibm_kingston | [0, 1, 2, 3] | 37 / 6 | 0.0049-0.0229 | 0.0010-0.0060 | 0.630 | 0/2, p=0.5 | 2 (p=0.854) | -0.0087 [-0.0121, -0.0055] |
| ibm_fez | [0, 1, 2, 3] | 37 / 6 | 0.0039-0.0293 | 0.0016-0.0072 | 0.640 | 2/3, p=1 | 5 (p=0.174) | -0.0152 [-0.0208, -0.0095] |
| fake_marrakesh | [0, 1, 2, 3] | 37 / 6 | 0.0042-0.0112 | 0.0015-0.0032 | 0.660 | 1/0, p=1 | 1 (p=0.972) | -0.0045 [-0.0078, -0.0012] |
| fake_fez | [0, 1, 2, 3] | 37 / 6 | 0.0049-0.0159 | 0.0030-0.0102 | 0.640 | 1/2, p=1 | 3 (p=0.63) | -0.0092 [-0.0135, -0.0049] |

Paired differences in mean shift (same inputs; Marrakesh averaged over its runs), 95% bootstrap CI over inputs:

| Comparison | Difference [95% CI] |
|---|---|
| ibm_kingston - ibm_marrakesh | -0.0037 [-0.0071, -0.0002] |
| ibm_fez - ibm_marrakesh | -0.0101 [-0.0145, -0.0058] |
| ibm_fez - ibm_kingston | -0.0065 [-0.0113, -0.0015] |
| fake_fez - ibm_fez | +0.0060 [+0.0015, +0.0104] |
| fake_marrakesh - ibm_marrakesh | +0.0005 [-0.0029, +0.0040] |
