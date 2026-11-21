"""Summarise EKF results + make figures.  usage: python analyze.py real|synth"""
import numpy as np, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

METHODS = ["baseline_td", "baseline", "heuristic", "mlp", "lstm", "oracle"]
NAMES = {"baseline_td": "Original EKF (2b replay)", "baseline": "EKF, continuous updates", "heuristic": "Heuristic", "mlp": "MLP", "lstm": "LSTM-MLP", "oracle": "Oracle labels"}
COL = {"baseline_td": "#c9c8c2", "baseline": "#8a8984", "heuristic": "#2a78d6", "mlp": "#eb6834", "lstm": "#1baf7a", "oracle": "#eda100"}
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": "#ecebe7",
                     "axes.axisbelow": True})


def load(exp):
    rows = []
    for m in METHODS:
        p = ROOT / "results" / f"ekf_{exp}" / f"{m}.json"
        if p.exists():
            rows += json.load(open(p))
    return rows


def add_damage(rows, exp):
    """Synthetic only: RMS / final distance between a run on slipped data and the clean-data baseline run
    (same sequence, same timing offset). Isolates slip-induced error from heading drift."""
    if exp != "synth":
        return rows
    tr = ROOT / "results" / "ekf_synth" / "traj"; cl = ROOT / "results" / "ekf_real" / "traj"
    for r in rows:
        a = np.load(tr / f"{r['method']}_{r['seq']}_{r['sig_max']}_{r['offset']:+.2f}.npz")
        clean = "baseline_td" if r["method"] == "baseline_td" else "baseline"     # same filter, clean data
        b = np.load(cl / f"{clean}_{r['seq']}_None_{r['offset']:+.2f}.npz")
        n = min(len(a["P"]), len(b["P"]))
        dP = a["P"][:n] - b["P"][:n]
        r["damage"] = float(np.sqrt(np.mean(np.sum(dP ** 2, axis=1))))
        r["damage_final"] = float(np.linalg.norm(dP[-1]))
    return rows


def select_sigma(rows, m):
    """Pick sig_max for method m by mean RPE over the TRAIN sequences (test never used)."""
    if m.startswith("baseline"):
        return None
    sig = sorted({r["sig_max"] for r in rows if r["method"] == m})
    key = "damage" if "damage" in rows[0] else "RPE"
    score = {s: np.mean([r[key] for r in rows if r["method"] == m and r["sig_max"] == s and r["seq"] in TRAIN]) for s in sig}
    return min(score, key=score.get)


def table(rows, seqs):
    out = {}
    for m in METHODS:
        if not any(r["method"] == m for r in rows):
            continue
        sm = select_sigma(rows, m)
        out[m] = {"sig_max": sm}
        for s in seqs + ["mean_test"]:
            ss = TEST if s == "mean_test" else [s]
            sel = [r for r in rows if r["method"] == m and r["sig_max"] == sm and r["seq"] in ss]
            out[m][s] = {k: (float(np.mean([r[k] for r in sel])), float(np.std([r[k] for r in sel]))) for k in KEYS(rows)}
    return out


def KEYS(rows):
    return ["ATE", "RPE", "drift"] + (["damage", "damage_final"] if "damage" in rows[0] else [])


def md(tab, exp):
    L = [f"## Experiment {'A: real LIO labels' if exp=='real' else 'B: synthetic slips'} (test = stair, trail)\n",
         f"mean ± std over 5 foot-timing offsets (−40…+40 ms); σ_max chosen on training sequences by {'slip damage' if exp=='synth' else 'RPE'}\n",
         "| Method | σ_max | Seq | ATE [m] | RPE /10 m [m] | Drift [%] |" + (" Slip damage RMS [m] | Slip damage final [m] |" if exp == "synth" else ""),
         "|---|---|---|---|---|---|" + ("---|---|" if exp == "synth" else "")]
    for m, d in tab.items():
        for s in TEST + ["mean_test"]:
            v = d[s]
            L.append(f"| {NAMES[m]} | {d['sig_max'] if d['sig_max'] else '—'} | {s} | {v['ATE'][0]:.2f} ± {v['ATE'][1]:.2f} | "
                     f"{v['RPE'][0]:.3f} ± {v['RPE'][1]:.3f} | {v['drift'][0]:.2f} ± {v['drift'][1]:.2f} |"
                     + (f" {v['damage'][0]:.2f} ± {v['damage'][1]:.2f} | {v['damage_final'][0]:.2f} ± {v['damage_final'][1]:.2f} |" if "damage" in v else ""))
    return "\n".join(L)


def fig_bars(tab, exp):
    ms = list(tab.keys())
    panels = [("ATE", "ATE [m]"), ("RPE", "RPE per 10 m [m]"), ("drift", "Drift [% of path]")]
    if exp == "synth":
        panels = [("damage", "Slip damage, RMS [m]")] + panels
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6))
    for ax, (k, lab) in zip(axes, panels):
        x = np.arange(len(TEST)); w = 0.8 / len(ms)
        for j, m in enumerate(ms):
            mu = [tab[m][s][k][0] for s in TEST]; sd = [tab[m][s][k][1] for s in TEST]
            xx = x - 0.4 + w / 2 + j * w
            ax.bar(xx, mu, w - 0.02, color=COL[m], label=NAMES[m], yerr=sd, ecolor=INK2, capsize=2, error_kw=dict(lw=0.8))
        ax.set_xticks(x); ax.set_xticklabels(TEST); ax.set_ylabel(lab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=len(ms), bbox_to_anchor=(0.5, 0.93))
    t = "Experiment A — real LIO labels" if exp == "real" else "Experiment B — synthetic slips injected"
    fig.suptitle(f"{t}: test sequences (error bars = std over 5 timing offsets)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    p = ROOT / "figures" / f"ekf_metrics_{exp}.png"; fig.savefig(p, dpi=160); plt.close(fig)
    return p


if __name__ == "__main__":
    exp = sys.argv[1]
    (ROOT / "figures").mkdir(exist_ok=True)
    rows = add_damage(load(exp), exp)
    tab = table(rows, SEQS)
    json.dump(tab, open(ROOT / "results" / f"summary_{exp}.json", "w"), indent=1)
    txt = md(tab, exp)
    open(ROOT / "results" / f"summary_{exp}.md", "w").write(txt)
    print(txt)
    print(fig_bars(tab, exp))
