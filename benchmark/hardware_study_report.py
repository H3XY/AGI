"""Analyse results/hardware_study/*.json (all sessions pooled per task and backend).

Per task and backend:
  - accuracy of each run with a Wilson 95% CI, and the ideal accuracy on the same inputs
  - flips vs the ideal prediction per run, the exact expected count under shot
    noise alone (binomial, rule: class 1 iff P > 0.5), and the exact
    Poisson-binomial tail P(flips >= observed) pooled over runs
  - mean signed shift of P(class 1) (backend minus ideal) per run, and pooled
    with a 95% bootstrap CI over inputs
  - run-to-run agreement of predictions (real device only)
  - calibration snapshot summary of the physical qubits used

Usage:  python -m benchmark.hardware_study_report
Writes results/hardware_study/report.md and report.json
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binom

from benchmark.uncertainty import wilson

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hardware_study")


def main():
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(OUT, "*_*_*.json"))):
        d = json.load(open(f))
        if "runs" in d:
            groups[(d["task"], d["backend"])].append(d)
    report, L = {}, ["# Hardware study: trained text classifiers on ibm_marrakesh and two noise models", ""]
    for (task, backend), ds in sorted(groups.items()):
        y = np.array(ds[0]["labels"])
        p = np.array(ds[0]["p_ideal"])
        assert all(d["test_indices"] == ds[0]["test_indices"] for d in ds), "sessions must use the same inputs"
        runs = [(d["label"], r) for d in ds for r in d["runs"]]
        P = np.array([r["p"] for _, r in runs])
        shots = ds[0]["shots"]
        ideal_pred = (p > 0.5).astype(int)
        q = np.where(p > 0.5, binom.cdf(shots // 2, shots, p), binom.sf(shots // 2, shots, p))
        flips = [int(np.sum((row > 0.5).astype(int) != ideal_pred)) for row in P]
        dist = np.array([1.0])
        for qi in np.tile(q, len(P)):
            dist = np.convolve(dist, [1 - qi, qi])
        shift_in = (P - p).mean(axis=0)
        rng = np.random.default_rng(0)
        bs = shift_in[rng.integers(0, len(p), (2000, len(p)))].mean(axis=1)
        accs = [float(np.mean((row > 0.5).astype(int) == y)) for row in P]
        preds = (P > 0.5).astype(int)
        agree = float(np.mean([np.mean(preds[i] == preds[j]) for i in range(len(P)) for j in range(i + 1, len(P))])) if len(P) > 1 else None
        r = {"inputs": int(len(y)), "shots": shots, "runs": len(P), "sessions": sorted({d["label"] for d in ds}),
             "ideal_accuracy": float(np.mean(ideal_pred == y)),
             "accuracy_per_run": accs, "wilson_ci_per_run": [wilson(int(round(a * len(y))), len(y)) for a in accs],
             "flips_per_run": flips, "expected_flips_per_run": float(q.sum()),
             "p_value_flips": float(dist[sum(flips):].sum()),
             "shift_per_run": [float(np.mean(row - p)) for row in P],
             "shift_pooled": float(shift_in.mean()),
             "shift_pooled_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
             "run_to_run_prediction_agreement": agree,
             "job_ids": [rr.get("job_id") for _, rr in runs if rr.get("job_id")],
             "finished_utc": [rr.get("finished_utc") for _, rr in runs]}
        cal = ds[0].get("calibration_at_start", {})
        if cal:
            ro = [v["readout_error"] for v in cal["qubits"].values() if v.get("readout_error") is not None]
            cz = list(cal.get("two_qubit_gates", {}).values())
            r["calibration"] = {"physical_qubits": cal.get("physical_qubits"), "compiled": cal.get("compiled"),
                                "readout_error_range": [min(ro), max(ro)] if ro else None,
                                "two_qubit_error_range": [min(cz), max(cz)] if cz else None}
        report[f"{task}/{backend}"] = r
        L += [f"## {task} on {backend} ({r['inputs']} inputs, {shots} shots, {r['runs']} run(s), sessions {', '.join(r['sessions'])})", "",
              f"- Ideal accuracy on these inputs: {r['ideal_accuracy']:.3f}",
              "- Accuracy per run (Wilson 95% CI): " + ", ".join(f"{a:.3f} [{c[0]:.3f}, {c[1]:.3f}]" for a, c in zip(accs, r["wilson_ci_per_run"])),
              f"- Flips vs ideal per run: {flips}; expected from shot noise alone: {r['expected_flips_per_run']:.2f} per run; "
              f"P(total >= {sum(flips)}) = {r['p_value_flips']:.3g}",
              f"- Mean signed shift of P(class 1): per run {np.round(r['shift_per_run'], 4).tolist()}; pooled {r['shift_pooled']:+.4f} "
              f"(95% CI [{r['shift_pooled_ci'][0]:+.4f}, {r['shift_pooled_ci'][1]:+.4f}])"]
        if agree is not None:
            L.append(f"- Run-to-run prediction agreement: {agree:.3f}")
        if "calibration" in r and r["calibration"]["readout_error_range"]:
            c = r["calibration"]
            L.append(f"- Qubits {c['physical_qubits']}; readout error {c['readout_error_range'][0]:.4f}-{c['readout_error_range'][1]:.4f}; "
                     f"two-qubit error {c['two_qubit_error_range'][0]:.4f}-{c['two_qubit_error_range'][1]:.4f}; compiled depth {c['compiled']['depth']}, "
                     f"{c['compiled']['two_qubit_gates']} two-qubit gates")
        if r["job_ids"]:
            L.append(f"- Job IDs: {', '.join(r['job_ids'])}")
        L.append("")
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2)
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
