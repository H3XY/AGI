"""Figure for the hardware study: accuracy per run (Wilson 95% CI) and pooled bias (95% CI).

Usage:  python -m benchmark.hardware_study_figure   (after hardware_study_report)
Writes results/hardware_study/figure.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hardware_study")
BACKENDS = [("ibm_marrakesh", "ibm_marrakesh (real)", "#2a78d6"),
            ("fake_marrakesh", "FakeMarrakesh (same device, simulated)", "#1baf7a"),
            ("fake_manila", "FakeManilaV2 (older 5-qubit device, simulated)", "#eb6834")]
TEXT, TEXT2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def main():
    rep = json.load(open(os.path.join(OUT, "report.json")))
    tasks = sorted({k.split("/")[0] for k in rep})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=170, gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT2, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)

    # (a) accuracy per run
    ax = axes[0]
    for ti, task in enumerate(tasks):
        ideal = rep[f"{task}/ibm_marrakesh"]["ideal_accuracy"]
        ax.plot([ti - 0.38, ti + 0.38], [ideal, ideal], color=TEXT, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        for bi, (key, label, color) in enumerate(BACKENDS):
            r = rep.get(f"{task}/{key}")
            if not r:
                continue
            n = len(r["accuracy_per_run"])
            for k, (a, ci) in enumerate(zip(r["accuracy_per_run"], r["wilson_ci_per_run"])):
                x = ti + (bi - 1) * 0.25 + (k - (n - 1) / 2) * 0.06
                ax.errorbar(x, a, yerr=[[a - ci[0]], [ci[1] - a]], fmt="o", color=color, markersize=5,
                            elinewidth=1.4, capsize=0, zorder=3, label=label if (ti == 0 and k == 0) else None)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([f"{t}\n(200 inputs)" for t in tasks], color=TEXT, fontsize=9)
    ax.set_ylabel("Accuracy per run (Wilson 95% CI)", color=TEXT2, fontsize=9)
    ax.set_title("(a) Accuracy; dashed = noise-free simulator", loc="left", color=TEXT, fontsize=10)

    # (b) pooled bias
    ax = axes[1]
    ax.axvline(0, color=TEXT2, linewidth=1)
    yt, yl = [], []
    row = 0
    for task in tasks:
        for key, label, color in BACKENDS:
            r = rep.get(f"{task}/{key}")
            if not r:
                continue
            m, (lo, hi) = r["shift_pooled"], r["shift_pooled_ci"]
            ax.errorbar(m, row, xerr=[[m - lo], [hi - m]], fmt="o", color=color, markersize=5, elinewidth=1.4, capsize=0, zorder=3)
            yt.append(row)
            yl.append(f"{task}, {key}")
            row += 1
        row += 0.6
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=7.5, color=TEXT)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean shift of P(class 1) vs noise-free (95% CI)", color=TEXT2, fontsize=9)
    ax.set_title("(b) Bias", loc="left", color=TEXT, fontsize=10)

    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=TEXT, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, "figure.png"), facecolor=SURFACE)
    print("Wrote results/hardware_study/figure.png")


if __name__ == "__main__":
    main()
