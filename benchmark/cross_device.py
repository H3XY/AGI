"""Cross-device replication: the CVE classifier on ibm_marrakesh, ibm_kingston and ibm_fez.

Uses the 100-input stratified sample of the Kingston/Fez runs (label `xdev`), which is a
subset of the 200 inputs of the Marrakesh study (same sampler, same seed), so every device
is compared on identical inputs against the same noise-free probabilities. Same model,
same 1,024 shots, preset pass manager with seed_transpiler=0 for each device.

Per device run: accuracy, exact McNemar test against noise-free correctness, changed
predictions vs the exact shot-noise expectation (Poisson-binomial tail), and mean output
shift with a 95% bootstrap CI over inputs. Between devices: paired difference in shift,
bootstrap CI over inputs. Noise models FakeFez and FakeMarrakesh on the same inputs.

Usage:  python -m benchmark.cross_device
Writes results/hardware_study/cross_device.{md,json}
"""
import json
import os

import numpy as np
from scipy.stats import binom, binomtest

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hardware_study")
SOURCES = [("ibm_marrakesh", ["pilot", "session1", "session2"]), ("ibm_kingston", ["xdev"]), ("ibm_fez", ["xdev"]),
           ("fake_marrakesh", ["session1"]), ("fake_fez", ["xdev"])]
B = 4000


def load(backend, label, idx):
    d = json.load(open(os.path.join(OUT, f"{label}_cve_{backend}.json")))
    pos = {t: k for k, t in enumerate(d["test_indices"])}
    sel = [pos[t] for t in idx]
    cal = d.get("calibration_at_start", {})
    return [np.array(r["p"])[sel] for r in d["runs"]], cal


def main():
    ref = json.load(open(os.path.join(OUT, "xdev_cve_ibm_kingston.json")))
    idx, y, p = ref["test_indices"], np.array(ref["labels"]), np.array(ref["p_ideal"])
    shots = ref["shots"]
    ideal_pred = (p > 0.5).astype(int)
    ideal_ok = ideal_pred == y
    q = np.where(p > 0.5, binom.cdf(shots // 2, shots, p), binom.sf(shots // 2, shots, p))
    dist = np.array([1.0])
    for qi in q:
        dist = np.convolve(dist, [1 - qi, qi])
    rng = np.random.default_rng(0)
    boot_idx = rng.integers(0, len(p), (B, len(p)))
    out = {"inputs": len(idx), "ideal_accuracy": float(ideal_ok.mean()), "expected_flips": float(q.sum()), "devices": {}}
    shifts = {}
    for backend, labels in SOURCES:
        rows, cals = [], []
        for lab in labels:
            runs, cal = load(backend, lab, idx)
            rows += runs
            cals.append(cal)
        res = []
        for r in rows:
            ok = ((r > 0.5).astype(int) == y)
            n_dev, n_ide = int(np.sum(ok & ~ideal_ok)), int(np.sum(~ok & ideal_ok))
            flips = int(np.sum((r > 0.5).astype(int) != ideal_pred))
            sh = r - p
            bs = sh[boot_idx].mean(axis=1)
            res.append({"accuracy": float(ok.mean()), "device_only_correct": n_dev, "ideal_only_correct": n_ide,
                        "mcnemar_p": float(binomtest(n_dev, n_dev + n_ide, 0.5).pvalue) if n_dev + n_ide else 1.0,
                        "flips": flips, "flips_p": float(dist[flips:].sum()),
                        "shift": float(sh.mean()), "shift_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]})
        shifts[backend] = np.mean([r - p for r in rows], axis=0)
        c = cals[0]
        ro = [v["readout_error"] for v in c.get("qubits", {}).values() if v.get("readout_error") is not None]
        tq = list(c.get("two_qubit_gates", {}).values())
        out["devices"][backend] = {"runs": res, "physical_qubits": c.get("physical_qubits"),
                                   "compiled": c.get("compiled", {}).get("depth"), "two_qubit_gates": c.get("compiled", {}).get("two_qubit_gates"),
                                   "readout_error_range": [min(ro), max(ro)] if ro else None,
                                   "two_qubit_error_range": [min(tq), max(tq)] if tq else None}
    pairs = [("ibm_kingston", "ibm_marrakesh"), ("ibm_fez", "ibm_marrakesh"), ("ibm_fez", "ibm_kingston"),
             ("fake_fez", "ibm_fez"), ("fake_marrakesh", "ibm_marrakesh")]
    out["shift_differences"] = {}
    for a, b in pairs:
        d = shifts[a] - shifts[b]
        bs = d[boot_idx].mean(axis=1)
        out["shift_differences"][f"{a} - {b}"] = {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    json.dump(out, open(os.path.join(OUT, "cross_device.json"), "w"), indent=2)

    L = ["# Cross-device replication: CVE classifier, 100 identical inputs", "",
         f"Noise-free accuracy on these inputs: {out['ideal_accuracy']:.3f}. Expected changed predictions from shot noise alone: "
         f"{out['expected_flips']:.2f} per run. Marrakesh rows are its three runs restricted to these 100 inputs.", "",
         "| Backend | Qubits | Depth / 2q gates | Readout err | 2q err | Accuracy | McNemar vs noise-free (dev-only/ideal-only, p) | Changed (p) | Mean shift [95% CI] |",
         "|---|---|---|---|---|---|---|---|---|"]
    for backend, dv in out["devices"].items():
        for k, r in enumerate(dv["runs"]):
            head = (f"| {backend}{' run ' + str(k + 1) if len(dv['runs']) > 1 else ''} | {dv['physical_qubits']} | {dv['compiled']} / {dv['two_qubit_gates']} | "
                    + (f"{dv['readout_error_range'][0]:.4f}-{dv['readout_error_range'][1]:.4f} | " if dv["readout_error_range"] else "- | ")
                    + (f"{dv['two_qubit_error_range'][0]:.4f}-{dv['two_qubit_error_range'][1]:.4f} | " if dv["two_qubit_error_range"] else "- | "))
            L.append(head + f"{r['accuracy']:.3f} | {r['device_only_correct']}/{r['ideal_only_correct']}, p={r['mcnemar_p']:.3g} | "
                     f"{r['flips']} (p={r['flips_p']:.3g}) | {r['shift']:+.4f} [{r['shift_ci'][0]:+.4f}, {r['shift_ci'][1]:+.4f}] |")
    L += ["", "Paired differences in mean shift (same inputs; Marrakesh averaged over its runs), 95% bootstrap CI over inputs:", "",
          "| Comparison | Difference [95% CI] |", "|---|---|"]
    for k, v in out["shift_differences"].items():
        L.append(f"| {k} | {v['diff']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] |")
    open(os.path.join(OUT, "cross_device.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
