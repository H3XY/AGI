# Circuit-design ablation (breast cancer, 4 PCA features unless noted, 3 seeds)

One factor changed at a time from the final design. Test accuracy is evaluated with exact probabilities.

| Variant | Encoding | Re-upload | Readout | Training | Optimizer | Qubits | Accuracy (mean ± std) | vs final |
|---|---|---|---|---|---|---|---|---|
| final | angle | yes | q0 | exact | lbfgs | 4 | 0.910 ± 0.011 | +0.000 |
| zz-encoding | zz | no | q0 | exact | lbfgs | 4 | 0.700 ± 0.032 | -0.211 |
| no-reupload | angle | no | q0 | exact | lbfgs | 4 | 0.887 ± 0.015 | -0.023 |
| parity-readout | angle | yes | parity | exact | lbfgs | 4 | 0.924 ± 0.005 | +0.014 |
| cobyla | angle | yes | q0 | exact | cobyla | 4 | 0.903 ± 0.003 | -0.008 |
| sampled-training | angle | yes | q0 | sampled | cobyla | 4 | 0.899 ± 0.012 | -0.012 |
| original | zz | no | parity | sampled | cobyla | 4 | 0.690 ± 0.051 | -0.220 |
| 2-qubits | angle | yes | q0 | exact | lbfgs | 2 | 0.916 ± 0.020 | +0.006 |
| 6-qubits | angle | yes | q0 | exact | lbfgs | 6 | 0.899 ± 0.015 | -0.012 |
