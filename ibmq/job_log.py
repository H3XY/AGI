"""Record the IBM Quantum jobs behind the published hardware results.

Lists recent jobs on the saved account (read-only: nothing is submitted) and
writes their IDs, backend, timestamps, status and quantum seconds, plus the
software versions and transpiler settings used, to results/ibm_jobs.json.

Usage:  python -m ibmq.job_log --limit 50
"""
import argparse
import json
import os
import platform
import warnings
from importlib.metadata import version

warnings.filterwarnings("ignore")

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--limit", type=int, default=50)
    args = a.parse_args()
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    jobs = []
    for j in service.jobs(limit=args.limit):
        m = j.metrics()
        ts = m.get("timestamps") or {}
        jobs.append({"job_id": j.job_id(), "backend": j.backend().name, "status": str(j.status()),
                     "created": j.creation_date.isoformat(), "running": ts.get("running"),
                     "finished": ts.get("finished"),
                     "quantum_seconds": (m.get("usage") or {}).get("quantum_seconds"),
                     "num_pubs": (m.get("num_pubs") if "num_pubs" in m else None)})
    jobs.sort(key=lambda r: r["created"])
    record = {
        "software": {pkg: version(pkg) for pkg in ["qiskit", "qiskit-ibm-runtime", "qiskit-machine-learning",
                                                    "qiskit-aer", "numpy", "scikit-learn", "sentence-transformers"]},
        "python": platform.python_version(),
        "transpiler": {"function": "generate_preset_pass_manager", "optimization_level": 1,
                       "layout_and_routing": "Qiskit defaults (no fixed initial layout, no seed_transpiler)",
                       "translation": "backend default (needs qiskit-ibm-runtime < 0.44 on Qiskit 1.x)"},
        "sampler": "qiskit_ibm_runtime.SamplerV2, default options except default_shots=1024; "
                   "no error mitigation or dynamical decoupling options were set",
        "jobs": jobs,
    }
    json.dump(record, open(os.path.join(RESULTS, "ibm_jobs.json"), "w"), indent=2)
    print(f"{len(jobs)} jobs written to results/ibm_jobs.json")
    for r in jobs:
        print(f"  {r['created'][:19]}  {r['backend']:14s} {r['status']:8s} qpu_s={r['quantum_seconds']}  {r['job_id']}")


if __name__ == "__main__":
    main()
