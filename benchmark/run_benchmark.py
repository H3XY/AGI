"""Quantum vs classical benchmark.

For each dataset: reduce features to one per qubit, train the QNN on an ideal
simulator, then measure test accuracy for
  - Quantum (ideal sim)       : the trained QNN on a noiseless simulator
  - Quantum (noisy IBM sim)   : the SAME trained weights on a simulated IBM device with realistic noise
  - Quantum (IBM hardware)    : optional, --hardware, needs IBM credentials. Runs on a
                                subset of the test set (quota), so it is reported in its own
                                table/chart next to the ideal simulator on the SAME samples.
  - Logistic regression / SVM : classical models on the SAME reduced features

Usage (from the repo root):
    python -m benchmark.run_benchmark                      # all datasets (~10 min)
    python -m benchmark.run_benchmark --datasets sentiment cve
    python -m benchmark.run_benchmark --hardware --hardware-samples 40
Writes results/benchmark.{json,md,png} (+ results/benchmark_hardware.png with --hardware).
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from benchmark.datasets import DATASETS
from core.features import make_angle_pipeline
from core.qnn_core import build_qnn, predict_proba, train_classifier

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

MODELS = [  # display order = fixed colour order in the chart
    "Quantum (ideal sim)",
    "Quantum (noisy IBM sim)",
    "Logistic regression",
    "SVM (RBF)",
]
HARDWARE_COLOR, IDEAL_COLOR = "#e87ba4", "#2a78d6"


def accuracy(y_true, p_class1):
    return float(np.mean((np.asarray(p_class1) > 0.5).astype(int) == np.asarray(y_true)))


def runtime_qnn(backend, num_qubits, reps, shots):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    return build_qnn(num_qubits=num_qubits, reps=reps, sampler=sampler, pass_manager=pm)


def load_split(spec, args):
    """-> X_fit, y_fit (for the reducer), X_tr, y_tr, X_te, y_te, description."""
    if spec["kind"] == "text":
        from core.text_tasks import TASKS, load_embeddings
        task = spec["task"]
        X_fit, y_fit = load_embeddings(task, "train")
        X_tr, y_tr = load_embeddings(task, "train", limit=args.train_samples, seed=args.seed)
        X_te, y_te = load_embeddings(task, "test", limit=args.test_samples, seed=args.seed)
        desc = (f"{TASKS[task]['description']}; reducer fitted on {len(y_fit)} training texts, "
                f"384-dim embeddings")
        return X_fit, y_fit, X_tr, y_tr, X_te, y_te, desc

    X, y, desc = spec["loader"]()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=args.seed)
    X_tr, y_tr = X_tr[: args.train_samples], y_tr[: args.train_samples]
    X_te, y_te = X_te[: args.test_samples], y_te[: args.test_samples]
    return X_tr, y_tr, X_tr, y_tr, X_te, y_te, desc


def run_dataset(name, args, hardware_backend=None):
    spec = DATASETS[name]
    num_qubits = spec["qubits"]
    X_fit, y_fit, X_tr, y_tr, X_te, y_te, desc = load_split(spec, args)
    print(f"\n=== {name}: {desc} -> {num_qubits} qubits")

    pipe = make_angle_pipeline(num_qubits, seed=args.seed, method=spec["reduce"]).fit(X_fit, y_fit)
    A_tr, A_te = pipe.transform(X_tr), pipe.transform(X_te)
    row = {"dataset": name, "description": desc, "qubits": num_qubits,
           "train_samples": len(y_tr), "test_samples": len(y_te), "accuracy": {}}

    # Classical baselines on the same reduced features
    for label, model in [("Logistic regression", LogisticRegression(max_iter=1000)),
                         ("SVM (RBF)", SVC())]:
        model.fit(A_tr, y_tr)
        row["accuracy"][label] = float(model.score(A_te, y_te))

    # Quantum: train on ideal simulator
    t0 = time.time()
    qnn = build_qnn(num_qubits=num_qubits, reps=args.reps)
    weights, losses = train_classifier(qnn, A_tr, y_tr, maxiter=args.maxiter, seed=args.seed)
    row["train_seconds"] = round(time.time() - t0, 1)
    row["final_loss"] = float(losses[-1]) if losses else None
    row["accuracy"]["Quantum (ideal sim)"] = accuracy(y_te, predict_proba(qnn, A_te, weights))

    # Same weights on a noisy simulated IBM device
    if not args.no_noise:
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2
        noisy = runtime_qnn(FakeManilaV2(), num_qubits, args.reps, args.shots)
        row["accuracy"]["Quantum (noisy IBM sim)"] = accuracy(y_te, predict_proba(noisy, A_te, weights))

    # Same weights on real hardware (subset, to save quota), compared with the
    # ideal simulator on exactly the same samples.
    if hardware_backend is not None:
        n = min(args.hardware_samples, len(y_te))
        hw = runtime_qnn(hardware_backend, num_qubits, args.reps, args.shots)
        p_ideal = predict_proba(qnn, A_te[:n], weights)
        p_hw = predict_proba(hw, A_te[:n], weights)
        row["hardware"] = {
            "backend": hardware_backend.name, "samples": int(n), "shots": args.shots,
            "ideal": accuracy(y_te[:n], p_ideal),
            "hardware": accuracy(y_te[:n], p_hw),
            # per-sample P(class 1), to see which answers noise flipped and by how much
            "labels": [int(v) for v in y_te[:n]],
            "p_ideal": [round(float(v), 4) for v in p_ideal],
            "p_hardware": [round(float(v), 4) for v in p_hw],
        }

    for k, v in row["accuracy"].items():
        print(f"  {k:26s} {v:.3f}")
    if "hardware" in row:
        h = row["hardware"]
        print(f"  On the first {h['samples']} test samples: ideal sim {h['ideal']:.3f}, "
              f"{h['backend']} {h['hardware']:.3f}")
    return row


def write_markdown(rows, path):
    present = [m for m in MODELS if any(m in r["accuracy"] for r in rows)]
    lines = ["# Quantum vs classical benchmark", "",
             "Test accuracy. All models see the same reduced features (one per qubit).", "",
             "| Dataset | Qubits | Train / test | " + " | ".join(present) + " |",
             "|---|---|---|" + "---|" * len(present)]
    for r in rows:
        cells = [f"{r['accuracy'][m]:.3f}" if m in r["accuracy"] else "-" for m in present]
        lines.append(f"| {r['dataset']} | {r['qubits']} | {r['train_samples']} / {r['test_samples']} | "
                     + " | ".join(cells) + " |")
    lines += ["", "Noisy IBM sim = the ideal-trained weights run on `FakeManilaV2` "
              "(a local simulation of a 5-qubit IBM device's noise)."]
    hw_rows = [r for r in rows if "hardware" in r]
    if hw_rows:
        lines += ["", "## Real IBM hardware", "",
                  "Same trained weights, same test samples (a subset, to save quantum time).", "",
                  "| Dataset | Device | Samples | Quantum (ideal sim) | Quantum (IBM hardware) | Difference |",
                  "|---|---|---|---|---|---|"]
        for r in hw_rows:
            h = r["hardware"]
            lines.append(f"| {r['dataset']} | {h['backend']} | {h['samples']} | {h['ideal']:.3f} | "
                         f"{h['hardware']:.3f} | {h['hardware'] - h['ideal']:+.3f} |")
        n = max(r["hardware"]["samples"] for r in hw_rows)
        lines += ["", f"With {n} samples, one sample is {1 / n:.3f} of accuracy, so a difference of "
                  "one or two samples can come from shot noise alone."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_hardware_chart(rows, path):
    """Ideal simulator vs real IBM hardware on the same test samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if "hardware" in r]
    text_primary, text_secondary, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    backend, n = rows[0]["hardware"]["backend"], rows[0]["hardware"]["samples"]
    series = [("Quantum (ideal sim)", "ideal", IDEAL_COLOR),
              (f"Quantum ({backend})", "hardware", HARDWARE_COLOR)]
    fig, ax = plt.subplots(figsize=(max(6, 1.7 * len(rows) + 2), 4.2), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    width = 0.36
    for j, (label, key, color) in enumerate(series):
        xs = [i - 0.2 + width * j + 0.02 for i in range(len(rows))]
        vals = [r["hardware"][key] for r in rows]
        bars = ax.bar(xs, vals, width=width * 0.9, color=color, label=label, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=7, color=text_secondary)
    ax.axhline(0.5, color=text_secondary, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{r['dataset']}\n({r['qubits']} qubits)" for r in rows], color=text_primary, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(f"Accuracy on the same test samples (up to {n})", color=text_secondary, fontsize=9)
    ax.tick_params(axis="y", colors=text_secondary, labelsize=8)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.set_title(f"Real quantum hardware vs ideal simulator ({backend})", loc="left",
                 color=text_primary, fontsize=11, pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
              fontsize=8, labelcolor=text_primary, handlelength=1, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def write_chart(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # dataviz reference palette, light mode, fixed slot order
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    text_primary, text_secondary, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    present = [m for m in MODELS if any(m in r["accuracy"] for r in rows)]

    n_ds, n_m = len(rows), len(present)
    width = 0.8 / n_m
    fig, ax = plt.subplots(figsize=(max(7, 2.4 * n_ds + 2), 4.6), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    for j, model in enumerate(present):
        xs, vals = [], []
        for i, r in enumerate(rows):
            if model in r["accuracy"]:
                xs.append(i - 0.4 + width * (j + 0.5))
                vals.append(r["accuracy"][model])
        bars = ax.bar(xs, vals, width=width * 0.88, color=colors[MODELS.index(model)],
                      label=model, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=7, color=text_secondary)

    ax.axhline(0.5, color=text_secondary, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.text(n_ds - 0.45, 0.505, "coin flip", fontsize=7, color=text_secondary, ha="right", va="bottom")
    ax.set_xticks(range(n_ds))
    ax.set_xticklabels([f"{r['dataset']}\n({r['qubits']} qubits)" for r in rows], color=text_primary, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test accuracy", color=text_secondary, fontsize=9)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", colors=text_secondary, labelsize=8)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.set_title("Quantum vs classical: test accuracy", loc="left", color=text_primary, fontsize=11, pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=min(n_m, 5), frameon=False,
              fontsize=8, labelcolor=text_primary, handlelength=1, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS),
                   choices=list(DATASETS))
    p.add_argument("--train-samples", type=int, default=300)
    p.add_argument("--test-samples", type=int, default=200)
    p.add_argument("--maxiter", type=int, default=60, help="L-BFGS-B iterations")
    p.add_argument("--reps", type=int, default=2, help="re-uploading layers")
    p.add_argument("--shots", type=int, default=1024, help="shots for noisy sim / hardware")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-noise", action="store_true", help="skip the noisy IBM simulation")
    p.add_argument("--hardware", action="store_true", help="also run on real IBM hardware")
    p.add_argument("--backend", help="IBM backend for --hardware (default: least busy)")
    p.add_argument("--hardware-samples", type=int, default=40)
    p.add_argument("--out", default="benchmark", help="output file name (no extension)")
    args = p.parse_args()

    hardware_backend = None
    if args.hardware:
        from ibmq.run_on_ibmq import get_backend
        hardware_backend = get_backend(args.backend, min_qubits=4)

    rows = [run_dataset(name, args, hardware_backend) for name in args.datasets]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    base = os.path.join(RESULTS_DIR, args.out)
    with open(base + ".json", "w") as f:
        json.dump(rows, f, indent=2)
    write_markdown(rows, base + ".md")
    write_chart(rows, base + ".png")
    if any("hardware" in r for r in rows):
        write_hardware_chart(rows, base + "_hardware.png")
    print(f"\nWrote {base}.json, {base}.md, {base}.png")


if __name__ == "__main__":
    main()
