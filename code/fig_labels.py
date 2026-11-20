"""Label figures: displacement histogram (real), score time series (real vs synthetic), label-prediction correlation."""
import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from analyze import COL, NAMES, INK, INK2, plt

if __name__ == "__main__":
    # 1. histogram of per-0.1 s stance-foot displacement
    fig, ax = plt.subplots(figsize=(6, 3.4))
    D = np.concatenate([np.load(ROOT / "features" / f"{s}.npz")["disp"].ravel() for s in TRAIN])
    D = D[~np.isnan(D)] * 100
    ax.hist(D, bins=np.arange(0, 8.01, 0.2), color="#2a78d6", edgecolor="#fcfcfb", linewidth=1)
    ax.axvline(2.0, color=INK2, lw=1, ls="--"); ax.text(2.1, ax.get_ylim()[1] * 0.9, "σ = 2 cm (score 0.61)", color=INK2, fontsize=8)
    ax.axvline(np.median(D), color=INK, lw=1); ax.text(np.median(D) - 0.1, ax.get_ylim()[1] * 0.75, f"median {np.median(D):.1f} cm", color=INK, fontsize=8, ha="right")
    ax.set_xlabel("planted-foot displacement in 0.1 s, from LIO [cm]"); ax.set_ylabel("count")
    ax.set_title("Real labels: one hump, weak tail; most of the spread is reference noise", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "label_histogram_real.png", dpi=160); plt.close(fig)

    # 2. score time series, front-left foot, 4 s of stair (test): real vs synthetic, label vs LSTM
    fig, axes = plt.subplots(1, 1, figsize=(10, 2.8)); axes = [axes]
    for ax, exp, sub in [(axes[0], "real", "")]:
        fd = np.load(ROOT / ("features" if exp == "real" else "features_synth") / "stair.npz")
        t = fd["t"] - fd["t"][0]; y = fd["y"][:, 0]; c = fd["contact"][:, 0]
        if exp == "synth":
            k = np.nanargmin(np.where(t > 20, y, np.nan)); t0 = t[k] - 2.0
        else:
            t0 = 60.0
        m = (t >= t0) & (t <= t0 + 4)
        ax.fill_between(t[m], 0, 1, where=c[m] > 0.5, color="#ecebe7", step="post", label="stance")
        lab = "LIO label" if exp == "real" else "true (injected) label"
        ax.plot(t[m], y[m], color=INK, lw=2, label=lab)
        for meth in ["heuristic", "lstm"]:
            p = ROOT / "results" / f"scores{sub}" / f"{meth}_stair.npy"
            if p.exists():
                sc = np.load(p)[:, 0]
                ax.plot(t[m], np.where(c[m] > 0.5, sc[m], np.nan), color=COL[meth], lw=2, label=NAMES[meth])
        ax.set_ylim(0, 1.05); ax.set_xlim(t0, t0 + 4); ax.set_ylabel("FL reliability"); ax.set_xlabel("time in stair sequence [s]")
        ax.set_title("Experiment A (real labels)" if exp == "real" else "Experiment B (synthetic slip)", fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left")

    fig.tight_layout(); fig.savefig(ROOT / "figures" / "scores_timeseries_stair.png", dpi=160); plt.close(fig)

    # 3. label-prediction correlation on the test sequences
    fig, axes = plt.subplots(1, 1, figsize=(5, 3.2)); axes = [axes]
    for ax, tag, title in [(axes[0], "", "Label prediction on test sequences (real LIO labels)")]:
        p = ROOT / "results" / f"label_prediction{tag}.json"
        if not p.exists():
            continue
        rep = json.load(open(p)); x = np.arange(len(TEST)); ms = ["heuristic", "mlp", "lstm"]
        for j, mth in enumerate(ms):
            v = [rep[s][mth]["r"] for s in TEST]
            b = ax.bar(x - 0.27 + j * 0.27, v, 0.25, color=COL[mth], label=NAMES[mth])
            for xi, vi in zip(x - 0.27 + j * 0.27, v):
                ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=7, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels(TEST); ax.set_title(title, fontsize=10, color=INK, loc="left")
    axes[0].set_ylabel("correlation with true label (r)"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "label_prediction_test.png", dpi=160); plt.close(fig)
    print("ok")
