"""Pre-registered confirmatory evaluation on untouched data (see results/confirmatory/PROTOCOL.md).

For each task and training seed 0, 1, 2, trains the frozen system exactly as in
benchmark/uncertainty.py and predicts on two sets:
  - the exploratory test split (as in the paper), to give per-seed paired statistics
  - the confirmatory split, never used before: Rotten Tomatoes `validation` (sentiment),
    NVD CVEs published May-Aug 2024 (cve)
Statistics: per-seed paired bootstrap CI and exact McNemar test; a two-level bootstrap
(seeds, then items) for the seed-averaged difference.

Usage:  python -m benchmark.confirmatory          (~45 min, CPU only)
        python -m benchmark.confirmatory --analyse-only   (recompute stats from saved predictions)
Writes results/confirmatory/{confirmatory.json, confirmatory.md, predictions_<task>.npz}
"""
import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "confirmatory")
CONFIRM_SPLIT = {"sentiment": "validation", "cve": "holdout"}
SEEDS = [0, 1, 2]
MODELS = ["Quantum (QNN)", "Logistic regression, CV-tuned", "SVM (RBF), CV-tuned", "Full embedding, logistic regression, CV-tuned"]
PRIMARY = ("Quantum (QNN)", "Logistic regression, CV-tuned")
B2 = 10000
B1 = 2000


def train_and_predict(task, maxiter=60):
    from core.features import make_angle_pipeline
    from core.qnn_core import build_qnn, predict_proba, train_classifier
    from core.text_tasks import load_embeddings

    X_full, y_full = load_embeddings(task, "train")
    sets = {"test": load_embeddings(task, "test"), "confirm": load_embeddings(task, CONFIRM_SPLIT[task])}
    pipe = make_angle_pipeline(4, seed=0, method="pls").fit(X_full, y_full)
    A = {k: pipe.transform(X) for k, (X, _) in sets.items()}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    full = GridSearchCV(LogisticRegression(max_iter=3000), {"C": [0.01, 0.1, 1, 10, 100]}, cv=cv).fit(X_full, y_full)

    preds = {s: {m: [] for m in MODELS} for s in sets}
    for seed in SEEDS:
        t0 = time.time()
        X_tr, y_tr = load_embeddings(task, "train", limit=600, seed=seed)
        A_tr = pipe.transform(X_tr)
        qnn = build_qnn(num_qubits=4, reps=2)
        w, _ = train_classifier(qnn, A_tr, y_tr, maxiter=maxiter, seed=seed, verbose=False)
        lr_t = GridSearchCV(LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10, 100]}, cv=cv).fit(A_tr, y_tr)
        svm_t = GridSearchCV(SVC(), {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.1, 1, 10]}, cv=cv).fit(A_tr, y_tr)
        for s, (X, _) in sets.items():
            preds[s]["Quantum (QNN)"].append((predict_proba(qnn, A[s], w) > 0.5).astype(int))
            preds[s]["Logistic regression, CV-tuned"].append(lr_t.predict(A[s]))
            preds[s]["SVM (RBF), CV-tuned"].append(svm_t.predict(A[s]))
            preds[s]["Full embedding, logistic regression, CV-tuned"].append(full.predict(X))
        print(f"  {task} seed {seed} trained in {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(os.path.join(OUT, f"predictions_{task}.npz"),
                        **{f"{s}|{m}": np.array(preds[s][m]) for s in sets for m in MODELS},
                        **{f"{s}|labels": sets[s][1] for s in sets})


def mcnemar(a_correct, b_correct):
    """Exact two-sided McNemar test on discordant pairs."""
    n01 = int(np.sum((a_correct == 1) & (b_correct == 0)))
    n10 = int(np.sum((a_correct == 0) & (b_correct == 1)))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, float(p)


def analyse(task):
    d = np.load(os.path.join(OUT, f"predictions_{task}.npz"))
    res = {}
    for s in ("test", "confirm"):
        y = d[f"{s}|labels"]
        C = {m: (d[f"{s}|{m}"] == y).astype(float) for m in MODELS}   # seeds x items
        n = len(y)
        r = {"items": int(n), "share_class1": float(y.mean()),
             "majority_baseline": float(max(y.mean(), 1 - y.mean())),
             "accuracy": {m: {"per_seed": C[m].mean(axis=1).tolist(), "mean": float(C[m].mean()),
                              "std": float(C[m].mean(axis=1).std(ddof=1))} for m in MODELS},
             "contrasts": {}}
        rng = np.random.default_rng(0)
        for other in MODELS[1:]:
            diff = C[PRIMARY[0]] - C[other]                                 # seeds x items
            per_seed = []
            for k, seed in enumerate(SEEDS):
                idx = rng.integers(0, n, (B1, n))
                bs = diff[k][idx].mean(axis=1)
                n01, n10, p = mcnemar(C[PRIMARY[0]][k], C[other][k])
                per_seed.append({"seed": seed, "diff": float(diff[k].mean()),
                                 "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                 "qnn_only_correct": n01, "other_only_correct": n10, "mcnemar_p": p})
            two = np.empty(B2)
            for b in range(B2):
                ss = rng.integers(0, len(SEEDS), len(SEEDS))
                ii = rng.integers(0, n, n)
                two[b] = diff[ss][:, ii].mean()
            r["contrasts"][other] = {"mean_diff": float(diff.mean()),
                                     "two_level_ci": [float(np.percentile(two, 2.5)), float(np.percentile(two, 97.5))],
                                     "per_seed": per_seed}
        res[s] = r
    return res


def verdict(task, c):
    lo, hi = c["two_level_ci"]
    if task == "sentiment":
        return "confirmed (QNN below)" if hi < 0 else ("reversed (QNN above)" if lo > 0 else "not confirmed (CI includes 0)")
    if -0.03 <= lo and hi <= 0.03:
        return "tie confirmed (CI within +/-3 points)"
    if lo > 0 or hi < 0:
        return "difference (CI excludes 0)"
    return "inconclusive"


def write(out):
    json.dump(out, open(os.path.join(OUT, "confirmatory.json"), "w"), indent=2)
    L = ["# Confirmatory evaluation (pre-registered: see PROTOCOL.md)", ""]
    for task, res in out.items():
        for s, label in (("confirm", f"CONFIRMATORY set ({CONFIRM_SPLIT[task]})"), ("test", "exploratory test set (as in the paper)")):
            r = res[s]
            L += [f"## {task}: {label}, {r['items']} items, {r['share_class1']:.1%} class 1 (majority baseline {r['majority_baseline']:.3f})", "",
                  "| Model | Seeds 0, 1, 2 | Mean ± std |", "|---|---|---|"]
            for m in MODELS:
                a = r["accuracy"][m]
                L.append(f"| {m} | {', '.join(f'{v:.3f}' for v in a['per_seed'])} | {a['mean']:.3f} ± {a['std']:.3f} |")
            L += ["", "| QNN minus | Mean diff | Two-level 95% CI (seeds, items) | Per seed: diff [paired CI], McNemar p |", "|---|---|---|---|"]
            for other, c in r["contrasts"].items():
                ps = "; ".join(f"{p['diff']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}], p={p['mcnemar_p']:.3g}" for p in c["per_seed"])
                L.append(f"| {other} | {c['mean_diff']:+.3f} | [{c['two_level_ci'][0]:+.3f}, {c['two_level_ci'][1]:+.3f}] | {ps} |")
            if s == "confirm":
                L += ["", f"**Pre-registered verdict:** {verdict(task, r['contrasts'][PRIMARY[1]])}"]
            L.append("")
    open(os.path.join(OUT, "confirmatory.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--tasks", nargs="+", default=["sentiment", "cve"])
    a.add_argument("--analyse-only", action="store_true")
    args = a.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.analyse_only:
        for t in args.tasks:
            train_and_predict(t)
    write({t: analyse(t) for t in args.tasks})


if __name__ == "__main__":
    main()
