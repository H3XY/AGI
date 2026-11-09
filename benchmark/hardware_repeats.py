"""Is a hardware accuracy drop shot noise or a systematic device effect?

Trains a dataset's QNN exactly as the benchmark does (deterministic), then runs
the same test samples on an IBM device several times, saving every sample's
P(class 1). For each sample it compares
  - observed: how often the hardware answer differs from the ideal answer
  - expected: how often 1,024-shot sampling ALONE would flip it, given the
    ideal probability p: P(flip) = Phi(-|p - 0.5| / sqrt(p (1 - p) / shots))
More flips than shot noise predicts, or the same samples flipping every run,
points to systematic hardware noise (e.g. a bias toward one class).

Usage (from the repo root; uses IBM quantum time, ~13 s per repeat):
    python -m benchmark.hardware_repeats --dataset cve --repeats 3
    python -m benchmark.hardware_repeats --dataset cve --repeats 3 --fake   # local noise model
Writes results/hardware_repeats_<dataset>.json and .md.
"""
import argparse
import json
import math
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from benchmark.datasets import DATASETS  # noqa: E402
from benchmark.run_benchmark import RESULTS_DIR, load_split, runtime_qnn  # noqa: E402
from core.features import make_angle_pipeline  # noqa: E402
from core.qnn_core import build_qnn, predict_proba, train_classifier  # noqa: E402


def shot_flip_probability(p, shots):
    """Chance that sampling `shots` times from ideal probability p lands on the other side of 0.5."""
    sd = math.sqrt(max(p * (1 - p), 1e-12) / shots)
    return 0.5 * math.erfc(abs(p - 0.5) / sd / math.sqrt(2))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--dataset", choices=list(DATASETS), default="cve")
    a.add_argument("--repeats", type=int, default=3)
    a.add_argument("--samples", type=int, default=40)
    a.add_argument("--shots", type=int, default=1024)
    a.add_argument("--backend", help="IBM backend (default: least busy)")
    a.add_argument("--fake", action="store_true", help="use the local FakeManilaV2 noise model instead")
    # must match the benchmark defaults so the trained model is identical
    a.add_argument("--train-samples", type=int, default=300)
    a.add_argument("--test-samples", type=int, default=200)
    a.add_argument("--maxiter", type=int, default=60)
    a.add_argument("--reps", type=int, default=2)
    a.add_argument("--seed", type=int, default=0)
    args = a.parse_args()

    spec = DATASETS[args.dataset]
    X_fit, y_fit, X_tr, y_tr, X_te, y_te, desc = load_split(spec, args)
    pipe = make_angle_pipeline(spec["qubits"], seed=args.seed, method=spec["reduce"]).fit(X_fit, y_fit)
    A_tr, A_te = pipe.transform(X_tr), pipe.transform(X_te)
    n = min(args.samples, len(y_te))
    A_te, y_te = A_te[:n], y_te[:n]

    print(f"Training {args.dataset} ({desc})...")
    qnn = build_qnn(num_qubits=spec["qubits"], reps=args.reps)
    weights, _ = train_classifier(qnn, A_tr, y_tr, maxiter=args.maxiter, seed=args.seed, verbose=False)
    p_ideal = predict_proba(qnn, A_te, weights)
    ideal_pred = (p_ideal > 0.5).astype(int)

    if args.fake:
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2
        backend = FakeManilaV2()
    else:
        from ibmq.run_on_ibmq import get_backend
        backend = get_backend(args.backend, min_qubits=spec["qubits"])
    hw_qnn = runtime_qnn(backend, spec["qubits"], args.reps, args.shots)

    runs = []
    for r in range(args.repeats):
        print(f"Repeat {r + 1}/{args.repeats} on {backend.name}...", flush=True)
        runs.append(predict_proba(hw_qnn, A_te, weights))
    P = np.array(runs)                       # repeats x samples
    hw_pred = (P > 0.5).astype(int)
    flips = hw_pred != ideal_pred            # repeats x samples

    expected = np.array([shot_flip_probability(p, args.shots) for p in p_ideal])
    shift = P - p_ideal                      # hardware minus ideal probability
    per_run_acc = [float(np.mean(hp == y_te)) for hp in hw_pred]
    flip_rate = flips.mean(axis=0)           # per sample, across repeats

    summary = {
        "dataset": args.dataset, "backend": backend.name, "samples": int(n),
        "shots": args.shots, "repeats": args.repeats,
        "ideal_accuracy": float(np.mean(ideal_pred == y_te)),
        "hardware_accuracy_per_run": per_run_acc,
        "flips_per_run": [int(f.sum()) for f in flips],
        "expected_flips_per_run_from_shot_noise": float(expected.sum()),
        "samples_flipped_in_every_run": int(np.sum(flips.all(axis=0))),
        "samples_flipped_at_least_once": int(np.sum(flips.any(axis=0))),
        "mean_shift_toward_class_1": float(shift.mean()),
        "mean_abs_shift": float(np.abs(shift).mean()),
        "mean_shift_toward_0.5": float(np.mean(np.abs(p_ideal - 0.5) - np.abs(P - 0.5))),
        "per_sample": [
            {"label": int(y_te[i]), "p_ideal": round(float(p_ideal[i]), 4),
             "p_hardware": [round(float(v), 4) for v in P[:, i]],
             "expected_flip_prob": round(float(expected[i]), 4), "flip_rate": float(flip_rate[i])}
            for i in range(n)
        ],
    }

    tag = f"{args.dataset}{'_fake' if args.fake else ''}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"hardware_repeats_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    s = summary
    lines = [f"# Hardware repeat study: {args.dataset} on {backend.name}", "",
             f"{n} test samples, {args.shots} shots, {args.repeats} repeats, same trained weights.", "",
             "| | Value |", "|---|---|",
             f"| Ideal simulator accuracy | {s['ideal_accuracy']:.3f} |",
             f"| Hardware accuracy per run | {', '.join(f'{v:.3f}' for v in per_run_acc)} |",
             f"| Answers flipped vs ideal, per run | {', '.join(map(str, s['flips_per_run']))} |",
             f"| Flips expected per run from shot noise alone | {s['expected_flips_per_run_from_shot_noise']:.2f} |",
             f"| Samples flipped in every run | {s['samples_flipped_in_every_run']} |",
             f"| Samples flipped at least once | {s['samples_flipped_at_least_once']} |",
             f"| Mean shift of P(class 1), hardware minus ideal | {s['mean_shift_toward_class_1']:+.4f} |",
             f"| Mean shrink toward 0.5 | {s['mean_shift_toward_0.5']:+.4f} |", "",
             "## Samples that flipped", "",
             "| Sample | Label | P ideal | P hardware (per run) | Shot-noise flip prob. | Flip rate |",
             "|---|---|---|---|---|---|"]
    for i, ps in enumerate(s["per_sample"]):
        if ps["flip_rate"] > 0:
            lines.append(f"| {i} | {ps['label']} | {ps['p_ideal']:.3f} | "
                         f"{', '.join(f'{v:.3f}' for v in ps['p_hardware'])} | "
                         f"{ps['expected_flip_prob']:.3f} | {ps['flip_rate']:.2f} |")
    with open(os.path.join(RESULTS_DIR, f"hardware_repeats_{tag}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
