# Quantum-Native AGI Prototype (IBM Quantum Ready)

This repository is a more complete prototype scaffold for a quantum-native AGI designed to be testable on IBM Quantum (Qiskit). It's intended for experimentation and education — **not** a production AGI.

## Contents
- `core/` - Core AGI files (agent loop, QNN, memory, utils)
- `modules/` - Perception, planner, simulation stubs
- `ibmq/` - Scripts to run on IBM Quantum backends (simulator + real devices)
- `requirements.txt` - Python dependencies
- `run_local.sh` - Quick local run script
- `README.md` - This file

## Quickstart
1. Create a virtualenv and install requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Set up IBM Quantum account:
   - Create an IBM Quantum account and get an API token.
   - Follow Qiskit instructions to save your credentials (`IBMQ.save_account`), or set environment variable `IBMQ_TOKEN`.
3. Run locally (simulator):
   ```bash
   python run.py
   ```
4. To run on IBM hardware, see `ibmq/run_on_ibmq.py` for examples.

## Notes & Next Steps
- The QNN modules use small numbers of qubits to remain runnable on current devices.
- Replace `Aer` simulator with `IBMQ` provider to run on real backends.
- Add persistence for memory and better classical-quantum hybrid orchestration for scaling.
