# Running on IBM Quantum

Two scripts can send the trained circuit to IBM Quantum via
[Qiskit IBM Runtime](https://quantum.cloud.ibm.com/docs). (The old `IBMQ` /
`qiskit-ibmq-provider` service has been retired.)

- `ibmq/run_on_ibmq.py`: score text with a trained classifier (sentiment or CVE), ideal
  simulator vs. IBM device side by side.
- `benchmark/run_benchmark.py --hardware`: add a real-hardware column to the benchmark.

`run_on_ibmq.py` uses the committed models in `models/` (retrain with
`python train.py --task sentiment|cve`). Add `--task cve` to score vulnerability
descriptions instead of opinions.

## 1. Credentials

Create an IBM Quantum Platform account at https://quantum.cloud.ibm.com, then
copy your API key and (optionally) your instance CRN. Either:

- set environment variables for the current shell:
  ```bash
  export IBM_QUANTUM_TOKEN="<your API key>"
  export IBM_QUANTUM_INSTANCE="<your instance CRN>"   # optional
  ```
- or save them once to `~/.qiskit/qiskit-ibm.json`:
  ```python
  from qiskit_ibm_runtime import QiskitRuntimeService
  QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
                                    token="<your API key>",
                                    instance="<your instance CRN>",  # optional
                                    set_as_default=True)
  ```

Never commit your token.

## 2. Run (from the repo root)

```bash
# Try it first with a local noisy simulation of an IBM device (no account needed)
python -m ibmq.run_on_ibmq "What a wonderful film" "A total snoozefest" --fake

# Real hardware: the least busy device your account can use
python -m ibmq.run_on_ibmq "What a wonderful film" "A total snoozefest"

# A specific device, fewer shots
python -m ibmq.run_on_ibmq "Great" --backend ibm_brisbane --shots 512

# Benchmark with a real-hardware column (40 test samples per dataset by default)
python -m benchmark.run_benchmark --hardware --hardware-samples 40
```

Real-hardware jobs wait in IBM's queue and use your account's quantum time.
Errors on hardware are raised rather than hidden behind a classical fallback.
