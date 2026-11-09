"""Uncertainty and robustness checks requested in review.

Part 1, text tasks (sentiment, cve), for training seeds 0, 1, 2. The seed picks
the 600-text QNN training subset and the QNN's initial weights. For each seed:
  - the QNN (exactly as in train.py)
  - logistic regression on the same 4 features, sklearn defaults (as reported)
  - logistic regression and RBF SVM on the same 4 features, tuned by 5-fold
    cross-validation on the same 600 training texts (never the test set)
  - logistic regression on the full 384-dim embedding, all training texts,
    defaults and CV-tuned
Reported: mean and std over seeds, a 95% bootstrap CI over test texts (seed 0),
and a paired bootstrap CI for QNN minus same-feature logistic regression.

Part 2, the CVE hardware repeats (results/hardware_repeats_cve*.json):
  - exact binomial flip probability per input under shot noise alone
    (prediction rule: class 1 iff the estimated P(class 1) > 0.5)
  - the exact Poisson-binomial distribution of total flips, and its tail
  - Wilson 95% CIs for each run's accuracy
  - bootstrap 95% CI over inputs for the mean signed shift (hardware - ideal)

Usage:  python -m benchmark.uncertainty            (~40 min, CPU only, no IBM access)
        python -m benchmark.uncertainty --part hardware   (seconds)
Writes results/uncertainty.json and results/uncertainty.md.
"""
import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from scipy.stats import binom  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from benchmark.run_benchmark import RESULTS_DIR  # noqa: E402

B = 2000  # bootstrap resamples


def boot_ci(correct, seed=0):
    """95% percentile bootstrap CI for accuracy from a 0/1 vector."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    stats = correct[rng.integers(0, n, (B, n))].mean(axis=1)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def paired_boot_ci(a, b, seed=0):
    """95% CI for mean(a) - mean(b), resampling test items jointly."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, (B, n))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def text_task(task, seeds, maxiter):
    from core.features import make_angle_pipeline
    from core.qnn_core import build_qnn, predict_proba, train_classifier
    from core.text_tasks import load_embeddings

    X_full, y_full = load_embeddings(task, "train")
    X_te, y_te = load_embeddings(task, "test")
    pipe = make_angle_pipeline(4, seed=0, method="pls").fit(X_full, y_full)  # same for every seed
    A_te = pipe.transform(X_te)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    # full-embedding baselines do not depend on the seed
    full_default = LogisticRegression(max_iter=2000).fit(X_full, y_full)
    full_tuned = GridSearchCV(LogisticRegression(max_iter=3000), {"C": [0.01, 0.1, 1, 10, 100]}, cv=cv).fit(X_full, y_full)
    out = {"task": task, "test_texts": int(len(y_te)), "seeds": seeds,
           "full_embedding": {
               "Logistic regression, defaults": float(full_default.score(X_te, y_te)),
               "Logistic regression, CV-tuned": float(full_tuned.score(X_te, y_te)),
               "CV-tuned C": full_tuned.best_params_["C"]},
           "per_seed": []}

    for seed in seeds:
        t0 = time.time()
        X_tr, y_tr = load_embeddings(task, "train", limit=600, seed=seed)
        A_tr = pipe.transform(X_tr)
        qnn = build_qnn(num_qubits=4, reps=2)
        w, _ = train_classifier(qnn, A_tr, y_tr, maxiter=maxiter, seed=seed, verbose=False)
        q_pred = (predict_proba(qnn, A_te, w) > 0.5).astype(int)
        lr = LogisticRegression(max_iter=1000).fit(A_tr, y_tr)
        lr_t = GridSearchCV(LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10, 100]}, cv=cv).fit(A_tr, y_tr)
        svm_t = GridSearchCV(SVC(), {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.1, 1, 10]}, cv=cv).fit(A_tr, y_tr)
        preds = {"Quantum (QNN)": q_pred, "Logistic regression, defaults": lr.predict(A_te),
                 "Logistic regression, CV-tuned": lr_t.predict(A_te), "SVM (RBF), CV-tuned": svm_t.predict(A_te)}
        row = {"seed": seed, "seconds": round(time.time() - t0, 1),
               "accuracy": {k: float(np.mean(v == y_te)) for k, v in preds.items()},
               "tuned_params": {"logreg": lr_t.best_params_, "svm": svm_t.best_params_}}
        if seed == seeds[0]:
            corr = {k: (v == y_te).astype(float) for k, v in preds.items()}
            row["bootstrap_ci"] = {k: boot_ci(c) for k, c in corr.items()}
            row["paired_ci_qnn_minus_logreg_defaults"] = paired_boot_ci(corr["Quantum (QNN)"], corr["Logistic regression, defaults"])
            row["paired_ci_qnn_minus_logreg_tuned"] = paired_boot_ci(corr["Quantum (QNN)"], corr["Logistic regression, CV-tuned"])
        out["per_seed"].append(row)
        print(f"  {task} seed {seed}: " + ", ".join(f"{k} {v:.3f}" for k, v in row["accuracy"].items()), flush=True)

    models = out["per_seed"][0]["accuracy"].keys()
    out["summary"] = {m: {"mean": float(np.mean([r["accuracy"][m] for r in out["per_seed"]])),
                          "std": float(np.std([r["accuracy"][m] for r in out["per_seed"]]))} for m in models}
    return out


def hardware_part():
    res = {}
    for tag in ["cve", "cve_fake"]:
        path = os.path.join(RESULTS_DIR, f"hardware_repeats_{tag}.json")
        if not os.path.exists(path):
            continue
        h = json.load(open(path))
        shots, runs = h["shots"], h["repeats"]
        p = np.array([s["p_ideal"] for s in h["per_sample"]])
        P = np.array([s["p_hardware"] for s in h["per_sample"]]).T  # runs x inputs
        y = np.array([s["label"] for s in h["per_sample"]])
        thr = shots // 2  # class 1 iff k/shots > 0.5, i.e. k >= thr + 1
        # exact P(prediction differs from the ideal prediction) under binomial shot noise
        q = np.where(p > 0.5, binom.cdf(thr, shots, p), binom.sf(thr, shots, p))
        # exact Poisson-binomial distribution of total flips over all runs
        dist = np.array([1.0])
        for qi in np.tile(q, runs):
            dist = np.convolve(dist, [1 - qi, qi])
        observed = int(sum(h["flips_per_run"]))
        ideal_pred = (p > 0.5).astype(int)
        shift = (P - p).mean(axis=0)  # per input, mean over runs
        rng = np.random.default_rng(0)
        bs = shift[rng.integers(0, len(shift), (B, len(shift)))].mean(axis=1)
        res[tag] = {
            "backend": h["backend"], "inputs": int(len(p)), "shots": shots, "runs": runs,
            "expected_flips_per_run_exact": float(q.sum()),
            "observed_flips": observed,
            "p_value_at_least_observed": float(dist[observed:].sum()),
            "accuracy_per_run": [float(np.mean(((r > 0.5).astype(int)) == y)) for r in P],
            "wilson_ci_per_run": [wilson(int(np.sum(((r > 0.5).astype(int)) == y)), len(y)) for r in P],
            "ideal_accuracy": float(np.mean(ideal_pred == y)),
            "mean_signed_shift": float(shift.mean()),
            "mean_signed_shift_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "inputs_with_abs_p_minus_half_below_0.05": int(np.sum(np.abs(p - 0.5) < 0.05)),
        }
        r = res[tag]
        print(f"  {tag}: expected {r['expected_flips_per_run_exact']:.2f} flips/run (exact), observed {observed} "
              f"over {runs} runs, P(X >= {observed}) = {r['p_value_at_least_observed']:.3f}; "
              f"shift {r['mean_signed_shift']:+.4f} CI {np.round(r['mean_signed_shift_ci'], 4).tolist()}")
    # the first benchmark run on the same 40 inputs (3 flips) had no per-input probabilities saved;
    # it used the same model and inputs, so it adds one more run under the same null
    if "cve" in res:
        h = json.load(open(os.path.join(RESULTS_DIR, "hardware_repeats_cve.json")))
        p = np.array([s["p_ideal"] for s in h["per_sample"]])
        thr = h["shots"] // 2
        q = np.where(p > 0.5, binom.cdf(thr, h["shots"], p), binom.sf(thr, h["shots"], p))
        dist = np.array([1.0])
        for qi in np.tile(q, h["repeats"] + 1):
            dist = np.convolve(dist, [1 - qi, qi])
        obs = int(sum(h["flips_per_run"])) + 3
        res["cve"]["including_first_benchmark_run"] = {"runs": h["repeats"] + 1, "observed_flips": obs,
                                                        "expected_flips": float(q.sum() * (h["repeats"] + 1)),
                                                        "p_value_at_least_observed": float(dist[obs:].sum())}
        print(f"  cve incl. first run: observed {obs}, expected {q.sum() * (h['repeats'] + 1):.2f}, "
              f"P(X >= {obs}) = {dist[obs:].sum():.3f}")
    return res


def write_md(out, path):
    L = ["# Uncertainty and robustness checks", ""]
    for t in out.get("text", []):
        L += [f"## {t['task']} ({t['test_texts']} test texts, seeds {t['seeds']})", "",
              "| Model (same 4 features unless noted) | Mean ± std over seeds | 95% bootstrap CI (seed 0) |", "|---|---|---|"]
        ci = t["per_seed"][0]["bootstrap_ci"]
        for m, s in t["summary"].items():
            L.append(f"| {m} | {s['mean']:.3f} ± {s['std']:.3f} | [{ci[m][0]:.3f}, {ci[m][1]:.3f}] |")
        fe = t["full_embedding"]
        L.append(f"| Logistic regression, full embedding, defaults | {fe['Logistic regression, defaults']:.3f} | |")
        L.append(f"| Logistic regression, full embedding, CV-tuned (C={fe['CV-tuned C']}) | {fe['Logistic regression, CV-tuned']:.3f} | |")
        s0 = t["per_seed"][0]
        L += ["", f"Paired 95% CI, QNN minus logistic regression (seed 0): defaults "
              f"[{s0['paired_ci_qnn_minus_logreg_defaults'][0]:+.3f}, {s0['paired_ci_qnn_minus_logreg_defaults'][1]:+.3f}], "
              f"CV-tuned [{s0['paired_ci_qnn_minus_logreg_tuned'][0]:+.3f}, {s0['paired_ci_qnn_minus_logreg_tuned'][1]:+.3f}].", ""]
    for tag, r in out.get("hardware", {}).items():
        L += [f"## Hardware repeats: {tag} ({r['backend']}, {r['inputs']} inputs, {r['shots']} shots, {r['runs']} runs)", "",
              f"- Expected flips per run from shot noise alone (exact binomial): {r['expected_flips_per_run_exact']:.2f}",
              f"- Observed flips: {r['observed_flips']}; P(at least that many | shot noise only) = {r['p_value_at_least_observed']:.3f}",
              f"- Accuracy per run: " + ", ".join(f"{a:.3f} [{c[0]:.3f}, {c[1]:.3f}]" for a, c in zip(r['accuracy_per_run'], r['wilson_ci_per_run'])) + " (Wilson 95% CI)",
              f"- Mean signed shift in P(class 1), hardware minus ideal: {r['mean_signed_shift']:+.4f}, 95% CI [{r['mean_signed_shift_ci'][0]:+.4f}, {r['mean_signed_shift_ci'][1]:+.4f}]"]
        if "including_first_benchmark_run" in r:
            f = r["including_first_benchmark_run"]
            L.append(f"- Including the first benchmark run ({f['runs']} runs): {f['observed_flips']} flips vs {f['expected_flips']:.2f} expected, P = {f['p_value_at_least_observed']:.3f}")
        L.append("")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--part", choices=["all", "text", "hardware"], default="all")
    a.add_argument("--tasks", nargs="+", default=["sentiment", "cve"])
    a.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a.add_argument("--maxiter", type=int, default=60)
    args = a.parse_args()
    path = os.path.join(RESULTS_DIR, "uncertainty.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    if args.part in ("all", "hardware"):
        print("Hardware repeats:")
        out["hardware"] = hardware_part()
    if args.part in ("all", "text"):
        out["text"] = [text_task(t, args.seeds, args.maxiter) for t in args.tasks]
    json.dump(out, open(path, "w"), indent=2)
    write_md(out, os.path.join(RESULTS_DIR, "uncertainty.md"))
    print("Wrote results/uncertainty.json and .md")


if __name__ == "__main__":
    main()
