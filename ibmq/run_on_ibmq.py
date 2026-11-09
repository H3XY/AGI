"""Score text with a trained quantum classifier on IBM Quantum.

Runs the same trained circuit three ways so you can see what hardware noise does:
ideal local simulator vs. a simulated noisy IBM device (--fake) or a real one.

Usage (from the repo root):
    python -m ibmq.run_on_ibmq --fake                                   # built-in examples, no account
    python -m ibmq.run_on_ibmq "What a wonderful film" "Total snoozefest"   # real device
    python -m ibmq.run_on_ibmq --task cve --fake                        # CVE severity examples
    python -m ibmq.run_on_ibmq "Great" --backend ibm_brisbane --shots 512

Needs a trained model (python train.py --task <task>). Credentials, checked in order:
    1. IBM_QUANTUM_TOKEN environment variable (plus optional IBM_QUANTUM_INSTANCE, a CRN)
    2. An account saved earlier with QiskitRuntimeService.save_account(...)
See ibmq/README_IBMQ.md.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Allow `python ibmq/run_on_ibmq.py` as well as `python -m ibmq.run_on_ibmq`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2  # noqa: E402


def get_service():
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE")
    if token:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
    try:
        return QiskitRuntimeService()  # uses a saved account
    except Exception as e:
        raise SystemExit(
            "No IBM Quantum credentials found. Set IBM_QUANTUM_TOKEN or save an account "
            "with QiskitRuntimeService.save_account(...). See ibmq/README_IBMQ.md.\n"
            f"Details: {e}"
        )


def get_backend(name=None, fake=False, min_qubits=2):
    if fake:
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2

        backend = FakeManilaV2()
        print("Using local fake backend (simulated IBM noise):", backend.name)
        return backend

    service = get_service()
    if name:
        backend = service.backend(name)
    else:
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)
    print("Selected backend:", backend.name)
    return backend


def runtime_sampler(backend, shots):
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    return sampler, generate_preset_pass_manager(backend=backend, optimization_level=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("texts", nargs="*", help="texts to score (default: the task's examples)")
    p.add_argument("--task", default="sentiment", help="sentiment or cve")
    p.add_argument("--backend", help="IBM backend name (default: least busy device)")
    p.add_argument("--fake", action="store_true", help="simulate an IBM device locally; no account needed")
    p.add_argument("--shots", type=int, default=1024)
    args = p.parse_args()

    from core.text_classifier import TextClassifier
    from core.text_tasks import TASKS
    if args.task not in TASKS:
        raise SystemExit(f"unknown task {args.task!r}; choose from {', '.join(TASKS)}")
    texts = args.texts or TASKS[args.task]["examples"]
    try:
        model = TextClassifier(args.task)
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    backend = get_backend(args.backend, fake=args.fake, min_qubits=model.num_qubits)
    remote_qnn = model.build_qnn_for(*runtime_sampler(backend, args.shots))

    ideal = model.predict(texts)
    # Errors on hardware are raised, not hidden behind a classical fallback.
    remote = model.predict(texts, qnn=remote_qnn)

    label = "noisy sim" if args.fake else backend.name
    print(f"\n{'text':45s} {'ideal sim':>10s} {label:>12s}")
    for s, a, b in zip(texts, ideal, remote):
        print(f"{s[:45]:45s} {a:10.2f} {b:12.2f}")
    print(f"\nValues are P({model.labels[1]}); above 0.5 means {model.labels[1]}.")


if __name__ == "__main__":
    main()
