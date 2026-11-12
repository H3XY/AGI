"""Dequantization check: classical Fourier surrogates of the QNN (see results/dequantization/PROTOCOL.md).

The QNN re-uploads each angle feature x_i twice with R_Y, so its output is a trigonometric
polynomial with integer frequencies in {-2..2} per feature. We fit classical models on that
exact basis (and on random subsets of it) with the same inputs and training texts as the QNN,
and compare accuracy on the exploratory test sets and the confirmatory sets.

Usage:  python -m benchmark.dequantize        (~5 min, CPU only; pre-registered grid)
        python -m benchmark.dequantize --posthoc   (post-hoc sensitivity: wider C grid,
                                                    because CV chose the grid edge C=0.001)
Writes results/dequantization/{dequantization.json, .md} (or dequantization_posthoc.*)
"""
import itertools
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402

from benchmark.confirmatory import CONFIRM_SPLIT, SEEDS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "dequantization")
CONF = os.path.join(ROOT, "results", "confirmatory")
KS = [8, 32, 128]
B2 = 10000


def frequencies(n=4, r=2):
    """Non-zero integer frequency vectors in {-r..r}^n, one of each +-pair."""
    out = []
    for w in itertools.product(range(-r, r + 1), repeat=n):
        w = np.array(w)
        nz = np.flatnonzero(w)
        if len(nz) and w[nz[0]] > 0:
            out.append(w)
    return np.array(out)


def fourier(A, W):
    Z = A @ W.T
    return np.hstack([np.cos(Z), np.sin(Z)])


def spectrum_check(task, W):
    """Fit the trained QNN's exact output with the Fourier basis; residual should be ~0."""
    from core.qnn_core import predict_proba
    from core.text_classifier import TextClassifier
    from core.qnn_core import build_qnn
    m = TextClassifier(task)
    qnn = build_qnn(num_qubits=4, reps=2)
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, (2000, 4))
    p = np.asarray(predict_proba(qnn, X, m.weights))
    F = np.hstack([np.ones((len(X), 1)), fourier(X, W)])
    coef, *_ = np.linalg.lstsq(F, p, rcond=None)
    resid = p - F @ coef
    # control: the same fit with frequencies limited to {-1..1} must NOT be exact
    W1 = frequencies(4, 1)
    F1 = np.hstack([np.ones((len(X), 1)), fourier(X, W1)])
    c1, *_ = np.linalg.lstsq(F1, p, rcond=None)
    return {"max_abs_residual": float(np.abs(resid).max()), "rms_residual": float(np.sqrt(np.mean(resid ** 2))),
            "control_r1_max_abs_residual": float(np.abs(p - F1 @ c1).max())}


def two_level(diff, rng):
    n = diff.shape[1]
    out = np.empty(B2)
    for b in range(B2):
        ss = rng.integers(0, diff.shape[0], diff.shape[0])
        ii = rng.integers(0, n, n)
        out[b] = diff[ss][:, ii].mean()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


GRID = [0.001, 0.01, 0.1, 1, 10, 100]
POSTHOC_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 0.001, 0.01, 0.1, 1, 10, 100]


def run_task(task, W):
    from core.features import make_angle_pipeline
    from core.text_tasks import load_embeddings

    X_full, y_full = load_embeddings(task, "train")
    pipe = make_angle_pipeline(4, seed=0, method="pls").fit(X_full, y_full)
    sets = {"test": load_embeddings(task, "test"), "confirm": load_embeddings(task, CONFIRM_SPLIT[task])}
    A = {s: pipe.transform(X) for s, (X, _) in sets.items()}
    q = np.load(os.path.join(CONF, f"predictions_{task}.npz"))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    grid = {"C": GRID}
    models = ["Fourier surrogate, full spectrum (624 features)"] + [f"Random Fourier subset, k={k}" for k in KS]
    preds = {s: {m: [] for m in models} for s in sets}
    chosen = {m: [] for m in models}
    rng_k = np.random.default_rng(1)
    subsets = {k: rng_k.choice(len(W), k, replace=False) for k in KS}
    for seed in SEEDS:
        X_tr, y_tr = load_embeddings(task, "train", limit=600, seed=seed)
        A_tr = pipe.transform(X_tr)
        for m, Wm in zip(models, [W] + [W[subsets[k]] for k in KS]):
            clf = GridSearchCV(LogisticRegression(max_iter=5000), grid, cv=cv).fit(fourier(A_tr, Wm), y_tr)
            chosen[m].append(clf.best_params_["C"])
            for s in sets:
                preds[s][m].append(clf.predict(fourier(A[s], Wm)))
    rng = np.random.default_rng(0)
    res = {"chosen_C": chosen}
    for s, (_, y) in sets.items():
        qc = (q[f"{s}|Quantum (QNN)"] == y).astype(float)
        lr = (q[f"{s}|Logistic regression, CV-tuned"] == y).astype(float)
        r = {"items": int(len(y)),
             "accuracy": {"Quantum (QNN)": {"per_seed": qc.mean(axis=1).tolist(), "mean": float(qc.mean())},
                          "Logistic regression on the 4 features, CV-tuned": {"per_seed": lr.mean(axis=1).tolist(), "mean": float(lr.mean())}},
             "qnn_minus": {}}
        for m in models:
            c = (np.array(preds[s][m]) == y).astype(float)
            r["accuracy"][m] = {"per_seed": c.mean(axis=1).tolist(), "mean": float(c.mean())}
            d = qc - c
            r["qnn_minus"][m] = {"diff": float(d.mean()), "two_level_ci": two_level(d, rng)}
        res[s] = r
    return res


def main():
    global GRID
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--posthoc", action="store_true")
    args = a.parse_args()
    name = "dequantization_posthoc" if args.posthoc else "dequantization"
    if args.posthoc:
        GRID = POSTHOC_GRID
    os.makedirs(OUT, exist_ok=True)
    W = frequencies()
    out = {"n_frequencies_nonzero": int(len(W)), "spectrum_check": {}, "tasks": {}}
    for task in ("sentiment", "cve"):
        out["spectrum_check"][task] = spectrum_check(task, W)
        out["tasks"][task] = run_task(task, W)
        print(task, "done", flush=True)
    json.dump(out, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
    L = ["# Dequantization check (" + ("POST-HOC sensitivity analysis, wider C grid " + str(GRID) if args.posthoc else "pre-registered: see PROTOCOL.md") + ")", "",
         f"Frequency basis: integer vectors in {{-2..2}}^4, {len(W)} non-zero up to sign (624 cos/sin features).", "",
         "## Spectrum check (trained seed-0 QNN, 2,000 random inputs)", "",
         "| Task | Max abs residual, {-2..2} basis | RMS residual | Control: max abs residual, {-1..1} basis |", "|---|---|---|---|"]
    for t, c in out["spectrum_check"].items():
        L.append(f"| {t} | {c['max_abs_residual']:.2e} | {c['rms_residual']:.2e} | {c['control_r1_max_abs_residual']:.3f} |")
    for t, r in out["tasks"].items():
        for s, label in (("confirm", f"confirmatory set ({CONFIRM_SPLIT[t]})"), ("test", "exploratory test set")):
            x = r[s]
            L += ["", f"## {t}: {label}, {x['items']} items", "", "| Model | Seeds 0, 1, 2 | Mean | QNN minus model [two-level 95% CI] |", "|---|---|---|---|"]
            for m, a in x["accuracy"].items():
                cmp = x["qnn_minus"].get(m)
                L.append(f"| {m} | {', '.join(f'{v:.3f}' for v in a['per_seed'])} | {a['mean']:.3f} | "
                         + (f"{cmp['diff']:+.3f} [{cmp['two_level_ci'][0]:+.3f}, {cmp['two_level_ci'][1]:+.3f}]" if cmp else "") + " |")
            if s == "confirm":
                ci = x["qnn_minus"]["Fourier surrogate, full spectrum (624 features)"]["two_level_ci"]
                verdict = "classically matched (CI at or below +0.01)" if ci[1] <= 0.01 else ("surrogate below the QNN (CI above 0)" if ci[0] > 0 else "inconclusive")
                L += ["", f"**{'Reading (same rule as the pre-registered one, applied post hoc)' if args.posthoc else 'Pre-registered reading'}:** {verdict}"]
        L += ["", f"CV-chosen C per seed: {r['chosen_C']}"]
    open(os.path.join(OUT, f"{name}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
