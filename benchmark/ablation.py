"""Which circuit-design choice mattered? A controlled ablation.

On breast cancer (4 PCA features, 300 training / 171 test samples), starting from
the final design (re-uploaded angle encoding, qubit-0 readout, exact
probabilities, L-BFGS-B), change ONE factor at a time, plus the original design
and two other qubit counts. Each variant runs for 3 seeds; the seed sets the
train/test split and the initial weights.

  final              angle encoding, re-uploaded x2, qubit-0 readout, exact, L-BFGS-B
  zz-encoding        ZZFeatureMap (once) + RealAmplitudes(reps=2), rest as final
  no-reupload        angle encoding once + RealAmplitudes(reps=2), rest as final
  zz-reupload        ZZFeatureMap re-uploaded x2, rest as final (with zz-encoding and
                     no-reupload, completes a 2x2 factorial of encoding x re-uploading)
  parity-readout     parity of all qubits instead of qubit 0
  cobyla             exact probabilities, but COBYLA instead of L-BFGS-B
  sampled-training   1,024-shot sampled probabilities, COBYLA (L-BFGS-B needs exact gradients)
  original           ZZFeatureMap, parity readout, sampled training, COBYLA
  2-qubits / 6-qubits  final design with 2 or 6 PCA features

Usage:  python -m benchmark.ablation        (~20 min, CPU only)
        python -m benchmark.ablation --variants zz-reupload   (run some variants, merge into the saved results)
Writes results/ablation.json and results/ablation.md.
"""
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import ParameterVector  # noqa: E402
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap  # noqa: E402
from qiskit.primitives import StatevectorSampler  # noqa: E402
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier  # noqa: E402
from qiskit_machine_learning.neural_networks import SamplerQNN  # noqa: E402
from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from benchmark.datasets import breast_cancer  # noqa: E402
from benchmark.run_benchmark import RESULTS_DIR  # noqa: E402
from core.features import make_angle_pipeline  # noqa: E402
from core.qnn_core import exact_sampler  # noqa: E402

VARIANTS = {
    "final":            dict(enc="angle", reupload=True,  readout="q0",     training="exact",   opt="lbfgs", n=4),
    "zz-encoding":      dict(enc="zz",    reupload=False, readout="q0",     training="exact",   opt="lbfgs", n=4),
    "no-reupload":      dict(enc="angle", reupload=False, readout="q0",     training="exact",   opt="lbfgs", n=4),
    "zz-reupload":      dict(enc="zz",    reupload=True,  readout="q0",     training="exact",   opt="lbfgs", n=4),
    "parity-readout":   dict(enc="angle", reupload=True,  readout="parity", training="exact",   opt="lbfgs", n=4),
    "cobyla":           dict(enc="angle", reupload=True,  readout="q0",     training="exact",   opt="cobyla", n=4),
    "sampled-training": dict(enc="angle", reupload=True,  readout="q0",     training="sampled", opt="cobyla", n=4),
    "original":         dict(enc="zz",    reupload=False, readout="parity", training="sampled", opt="cobyla", n=4),
    "2-qubits":         dict(enc="angle", reupload=True,  readout="q0",     training="exact",   opt="lbfgs", n=2),
    "6-qubits":         dict(enc="angle", reupload=True,  readout="q0",     training="exact",   opt="lbfgs", n=6),
}
SEEDS = [0, 1, 2]


def circuit(v, reps=2):
    n = v["n"]
    x = ParameterVector("x", n)
    qc = QuantumCircuit(n)
    weights = []
    layers = reps if v["reupload"] else 1
    for layer in range(layers):
        if v["enc"] == "zz":
            qc.compose(ZZFeatureMap(n, reps=1).assign_parameters(list(x)), inplace=True)
        else:
            for i in range(n):
                qc.ry(x[i], i)
        block = RealAmplitudes(n, reps=1 if v["reupload"] else reps, parameter_prefix=f"w{layer}")
        qc.compose(block, inplace=True)
        weights += list(block.parameters)
    qc.measure_all()
    return qc, list(x), weights


def run(name, v, seed):
    X, y, _ = breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    X_tr, y_tr = X_tr[:300], y_tr[:300]
    pipe = make_angle_pipeline(v["n"], seed=seed, method="pca").fit(X_tr, y_tr)
    A_tr, A_te = pipe.transform(X_tr), pipe.transform(X_te)
    qc, xi, ws = circuit(v)
    interpret = (lambda b: b & 1) if v["readout"] == "q0" else (lambda b: bin(b).count("1") % 2)
    sampler = exact_sampler() if v["training"] == "exact" else StatevectorSampler(default_shots=1024, seed=seed)
    qnn = SamplerQNN(circuit=qc, input_params=xi, weight_params=ws, sampler=sampler,
                     interpret=interpret, output_shape=2)
    opt = L_BFGS_B(maxiter=60) if v["opt"] == "lbfgs" else COBYLA(maxiter=150)
    init = np.random.default_rng(seed).uniform(-np.pi, np.pi, qnn.num_weights)
    clf = NeuralNetworkClassifier(qnn, loss="cross_entropy", optimizer=opt, one_hot=True, initial_point=init)
    clf.fit(A_tr, y_tr)
    # always evaluate with exact probabilities, so only training differs between variants
    eval_qnn = SamplerQNN(circuit=qc, input_params=xi, weight_params=ws, sampler=exact_sampler(),
                          interpret=interpret, output_shape=2)
    p1 = np.asarray(eval_qnn.forward(A_te, clf.weights))[:, 1]
    return float(np.mean((p1 > 0.5).astype(int) == y_te))


def main():
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = a.parse_args()
    path = os.path.join(RESULTS_DIR, "ablation.json")
    saved = json.load(open(path)) if os.path.exists(path) and set(args.variants) != set(VARIANTS) else {}
    out = {}
    for name, v in VARIANTS.items():
        if name not in args.variants:
            if name in saved:
                out[name] = saved[name]
            continue
        t0 = time.time()
        accs = [run(name, v, s) for s in SEEDS]
        out[name] = {"config": v, "accuracy_per_seed": accs, "mean": float(np.mean(accs)),
                     "std": float(np.std(accs)), "seconds": round(time.time() - t0, 1)}
        print(f"  {name:17s} {np.mean(accs):.3f} ± {np.std(accs):.3f}  {[round(a, 3) for a in accs]}", flush=True)
    json.dump(out, open(os.path.join(RESULTS_DIR, "ablation.json"), "w"), indent=2)
    base = out["final"]["mean"]
    L = ["# Circuit-design ablation (breast cancer, 4 PCA features unless noted, 3 seeds)", "",
         "One factor changed at a time from the final design. Test accuracy is evaluated with exact probabilities.", "",
         "| Variant | Encoding | Re-upload | Readout | Training | Optimizer | Qubits | Accuracy (mean ± std) | vs final |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, r in out.items():
        c = r["config"]
        L.append(f"| {name} | {c['enc']} | {'yes' if c['reupload'] else 'no'} | {c['readout']} | {c['training']} | "
                 f"{c['opt']} | {c['n']} | {r['mean']:.3f} ± {r['std']:.3f} | {r['mean'] - base:+.3f} |")
    open(os.path.join(RESULTS_DIR, "ablation.md"), "w").write("\n".join(L) + "\n")
    print("Wrote results/ablation.json and .md")


if __name__ == "__main__":
    main()
