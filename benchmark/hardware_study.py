"""Larger, repeated hardware study of the trained text classifiers.

For each task (sentiment, cve), take the model in models/<task>.joblib (the
paper's main models) and a stratified random sample of N test texts from the
full test split. Compile the circuit ONCE for the device with a fixed transpiler
seed, so every run uses the same physical qubits, then execute R separate jobs.
Record per-input probabilities, job IDs, timestamps and a calibration snapshot
of the qubits used. Local noise models (FakeMarrakesh, a snapshot of the same
device, and FakeManilaV2, an older 5-qubit device) run the same circuits for free.

A budget guard reads the instance's remaining quantum seconds before every
real-device job and stops if the estimated cost would exceed what is left
(minus a safety margin).

Usage (from the repo root):
    python -m benchmark.hardware_study --fake-only                 # local noise models only, free
    python -m benchmark.hardware_study --tasks cve --runs 1 --samples 200 --label pilot
    python -m benchmark.hardware_study --runs 3 --samples 200 --label session1
Writes results/hardware_study/<label>_<task>_<backend>.json
"""
import argparse
import json
import os
import time
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "hardware_study")
SECONDS_PER_CIRCUIT = 0.30   # initial estimate (40-input job took ~13 s); refined from the pilot
SECONDS_PER_JOB = 3.0


def sample_inputs(task, n, seed):
    from core.text_tasks import load_texts
    texts, y = load_texts(task, "test")
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[: int(round(n * np.mean(y == c)))] for c in (0, 1)])
    idx = np.sort(idx)[:n]
    return [texts[i] for i in idx], y[idx], idx


def calibration_snapshot(backend, isa_circuit):
    """Error rates and coherence of the physical qubits the compiled circuit uses."""
    layout = isa_circuit.layout.final_index_layout() if isa_circuit.layout else list(range(isa_circuit.num_qubits))
    used = sorted(set(layout))
    target = backend.target
    snap = {"physical_qubits": used, "qubits": {}, "two_qubit_gates": {}}
    for q in used:
        props = backend.qubit_properties(q) if hasattr(backend, "qubit_properties") else None
        meas = target["measure"][(q,)] if "measure" in target and (q,) in target["measure"] else None
        snap["qubits"][q] = {"t1_us": getattr(props, "t1", None) and props.t1 * 1e6,
                             "t2_us": getattr(props, "t2", None) and props.t2 * 1e6,
                             "readout_error": getattr(meas, "error", None)}
    for gate in ("cz", "ecr", "cx"):
        if gate in target:
            for qargs, p in target[gate].items():
                if qargs and set(qargs) <= set(used) and p is not None:
                    snap["two_qubit_gates"][f"{gate}{list(qargs)}"] = p.error
    counts = isa_circuit.count_ops()
    snap["compiled"] = {"depth": isa_circuit.depth(), "two_qubit_gates": int(isa_circuit.num_nonlocal_gates()),
                        "ops": {k: int(v) for k, v in counts.items()}}
    return snap


def remaining_seconds(service):
    u = service.usage()
    return u.get("usage_remaining_seconds"), u


def run_backend(model, A, backend, runs, shots, service=None, margin=30, sec_per_circuit=SECONDS_PER_CIRCUIT):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=0)
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    qnn = model.build_qnn_for(sampler, pm)   # compiled once here -> same qubits every run
    isa = qnn.circuit
    record = {"backend": backend.name, "shots": shots, "calibration_at_start": calibration_snapshot(backend, isa),
              "runs": []}
    for r in range(runs):
        if service is not None:
            left, usage = remaining_seconds(service)
            need = SECONDS_PER_JOB + sec_per_circuit * len(A)
            if left is None or left - need < margin:
                print(f"  budget guard: {left} s left, need ~{need:.0f} s + {margin} s margin; stopping")
                record["stopped_by_budget_guard"] = {"remaining_seconds": left, "estimated_need": need}
                break
            print(f"  run {r + 1}/{runs}: {left} s left, est. {need:.0f} s", flush=True)
        t0 = time.time()
        p = np.asarray(qnn.forward(A, model.weights))[:, 1]
        entry = {"run": r + 1, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "wall_seconds": round(time.time() - t0, 1), "p": [round(float(v), 5) for v in p]}
        if service is not None:
            j = service.jobs(limit=1)[0]
            m = j.metrics()
            entry.update({"job_id": j.job_id(), "quantum_seconds": (m.get("usage") or {}).get("quantum_seconds")})
        record["runs"].append(entry)
    return record


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--tasks", nargs="+", default=["sentiment", "cve"])
    a.add_argument("--samples", type=int, default=200)
    a.add_argument("--runs", type=int, default=3)
    a.add_argument("--shots", type=int, default=1024)
    a.add_argument("--backend", default="ibm_marrakesh")
    a.add_argument("--label", default="session1")
    a.add_argument("--fake-only", action="store_true")
    a.add_argument("--no-fake", action="store_true")
    a.add_argument("--seconds-per-circuit", type=float, default=SECONDS_PER_CIRCUIT)
    args = a.parse_args()

    from core.text_classifier import TextClassifier
    os.makedirs(OUT, exist_ok=True)
    service = backend = None
    if not args.fake_only:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.backend(args.backend)

    for task in args.tasks:
        model = TextClassifier(task)
        texts, y, idx = sample_inputs(task, args.samples, seed=0)
        A = model.encode(texts)
        p_ideal = model.predict(texts)
        base = {"task": task, "label": args.label, "inputs": len(y), "test_indices": idx.tolist(),
                "labels": y.tolist(), "p_ideal": [round(float(v), 5) for v in p_ideal],
                "ideal_accuracy": float(np.mean((p_ideal > 0.5) == y))}
        print(f"{task}: {len(y)} inputs, ideal accuracy {base['ideal_accuracy']:.3f}")
        targets = []
        if not args.no_fake:
            from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeMarrakesh
            targets += [(FakeMarrakesh(), None), (FakeManilaV2(), None)]
        if backend is not None:
            targets.append((backend, service))
        for b, svc in targets:
            rec = run_backend(model, A, b, args.runs, args.shots, service=svc, sec_per_circuit=args.seconds_per_circuit)
            out = dict(base, **rec)
            path = os.path.join(OUT, f"{args.label}_{task}_{b.name}.json")
            json.dump(out, open(path, "w"), indent=2)
            accs = [float(np.mean((np.array(r["p"]) > 0.5) == y)) for r in rec["runs"]]
            shift = [float(np.mean(np.array(r["p"]) - p_ideal)) for r in rec["runs"]]
            print(f"  {b.name}: accuracy per run {np.round(accs, 3).tolist()}, mean shift {np.round(shift, 4).tolist()}"
                  + (f", quantum s {[r.get('quantum_seconds') for r in rec['runs']]}" if svc else ""), flush=True)


if __name__ == "__main__":
    main()
