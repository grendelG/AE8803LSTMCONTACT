"""Run a method's scores through the EKF over several timing perturbations; return metrics."""
import numpy as np, json, sys
import run_ekf
from metrics import evaluate
from common import *

OFFSETS = [-0.04, -0.02, 0.0, 0.02, 0.04]   # foot-stream timing perturbations (Project 2b timing = 0)


def score_file(method, seq):
    return np.load(ROOT / "results" / "scores" / f"{method}_{seq}.npy")


def run_method(seq, scores=None, sig_max=None, offsets=OFFSETS):
    out = []
    for off in offsets:
        run_ekf.FOOT_DT = off
        T, P, Q = run_ekf.run(seq, scores=scores, sig_max=sig_max, fix_timing=True)
        r = evaluate(seq, T, P, Q); r["offset"] = off
        out.append(r)
    return out


def summarize(rs):
    return {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in ["ATE", "RPE", "drift"]}


if __name__ == "__main__":
    # oracle-label sweep of SIG_MAX on training sequences
    res = {}
    for sm in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        allr = []
        for s in TRAIN:
            y = np.load(ROOT / "features" / f"{s}.npz")["y"]
            allr += run_method(s, scores=np.nan_to_num(y, nan=1.0), sig_max=sm)
        res[sm] = summarize(allr)
        print("sig_max", sm, res[sm], flush=True)
    base = []
    for s in TRAIN:
        base += run_method(s)
    res["baseline"] = summarize(base)
    print("baseline", res["baseline"], flush=True)
    json.dump({str(k): v for k, v in res.items()}, open(ROOT / "results" / "sigmax_sweep_oracle_train.json", "w"), indent=1)
