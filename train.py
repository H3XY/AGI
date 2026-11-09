"""Train a quantum text classifier (used by the chat app and IBM script).

text -> 384-dim embedding (all-MiniLM-L6-v2)
     -> PLS down to one feature per qubit, scaled to [0, pi]
     -> QNN -> P(class 1)

Also trains two classical baselines so the quantum result has context:
  - logistic regression on the SAME reduced features (fair comparison)
  - logistic regression on the full embedding with all training data
    (what you'd actually ship)

Usage (from the repo root):
    python train.py --task sentiment
    python train.py --task cve
    python train.py --task cve --qubits 6 --train-samples 1000
Saves models/<task>.joblib and results/<task>.json.
"""
import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from core.features import make_angle_pipeline  # noqa: E402
from core.qnn_core import build_qnn, predict_proba, train_classifier  # noqa: E402
from core.text_encoder import EMBED_MODEL  # noqa: E402
from core.text_tasks import TASKS, load_embeddings  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=list(TASKS), default="sentiment")
    p.add_argument("--qubits", type=int, default=4)
    p.add_argument("--reps", type=int, default=2, help="re-uploading layers")
    p.add_argument("--train-samples", type=int, default=600,
                   help="texts the QNN trains on (quantum simulation is slow)")
    p.add_argument("--maxiter", type=int, default=60, help="L-BFGS-B iterations")
    p.add_argument("--reduce", choices=["pls", "pca"], default="pls",
                   help="pls = supervised reduction (default), pca = unsupervised")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    task = TASKS[args.task]

    X_full, y_full = load_embeddings(args.task, "train")
    X_tr, y_tr = load_embeddings(args.task, "train", limit=args.train_samples, seed=args.seed)
    X_te, y_te = load_embeddings(args.task, "test")
    print(f"{task['description']}: {len(y_tr)} training texts "
          f"(reducer fitted on all {len(y_full)}), {len(y_te)} test texts, {args.qubits} qubits")

    # The reducer is fitted on the whole training split, never the test split.
    pipe = make_angle_pipeline(args.qubits, seed=args.seed, method=args.reduce).fit(X_full, y_full)
    A_tr, A_te = pipe.transform(X_tr), pipe.transform(X_te)

    metrics = {"Majority class": float(max(y_te.mean(), 1 - y_te.mean()))}
    lr = LogisticRegression(max_iter=1000).fit(A_tr, y_tr)
    metrics["Logistic regression (same reduced features)"] = float(lr.score(A_te, y_te))
    lr_full = LogisticRegression(max_iter=2000).fit(X_full, y_full)
    metrics["Logistic regression (full embedding, all train data)"] = float(lr_full.score(X_te, y_te))

    t0 = time.time()
    qnn = build_qnn(num_qubits=args.qubits, reps=args.reps)
    weights, losses = train_classifier(qnn, A_tr, y_tr, maxiter=args.maxiter, seed=args.seed)
    train_seconds = round(time.time() - t0, 1)
    metrics["Quantum (ideal sim)"] = float(np.mean((predict_proba(qnn, A_te, weights) > 0.5) == y_te))

    print(f"\nTrained in {train_seconds}s. Test accuracy on {len(y_te)} texts:")
    for k, v in metrics.items():
        print(f"  {k:54s} {v:.3f}")

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    model_path = os.path.join(ROOT, "models", f"{args.task}.joblib")
    joblib.dump({
        "task": args.task, "labels": task["labels"], "num_qubits": args.qubits, "reps": args.reps,
        "weights": weights, "pipeline": pipe, "reduce": args.reduce,
        "embed_model": EMBED_MODEL, "metrics": metrics,
    }, model_path)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"{args.task}.json"), "w") as f:
        json.dump({"task": args.task, "args": vars(args), "train_seconds": train_seconds,
                   "test_texts": int(len(y_te)), "accuracy": metrics,
                   "loss_curve": [float(l) for l in losses]}, f, indent=2)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()
