"""Shared evaluation settings + a quick oracle-label sweep of sigma_max on the training sequences.

OFFSETS : foot-stream timing perturbations used for every EKF number (Project 2b timing = 0;
          the calibrated feet->IMU shift is +5 ms, so the set is centred on the true timing).
SIGMAS  : sigma_max grid for the reliability -> foothold-noise map. Must sit above run_ekf.SIG_MIN.
"""
import numpy as np, json, sys
from common import *

OFFSETS = [-0.04, -0.02, 0.0, 0.02, 0.04]
SIGMAS = [3e-2, 1e-1, 3e-1, 1.0]


def score_file(method, seq):
    return np.load(ROOT / "results" / "scores" / f"{method}_{seq}.npy")


def run_method(seq, scores=None, sig_max=None, offsets=OFFSETS):
    import run_ekf
    from metrics import evaluate
    out = []
    for off in offsets:
        run_ekf.FOOT_DT = off
        T, P, Q = run_ekf.run(seq, scores=scores, sig_max=sig_max, fix_timing=True)
        r = evaluate(seq, T, P, Q); r["offset"] = off
        out.append(r)
    return out


def summarize(rs):
    return {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in ["ATE", "RPE", "drift"]}


def _sweep(sm):
    allr = []
    for s in TRAIN:
        y = np.load(ROOT / "features" / f"{s}.npz")["y"]
        allr += run_method(s, scores=None if sm is None else np.nan_to_num(y, nan=1.0), sig_max=sm)
    return sm, summarize(allr)


if __name__ == "__main__":
    # oracle-label sweep of SIG_MAX on training sequences (continuous-update EKF)
    from multiprocessing import Pool
    with Pool(len(SIGMAS) + 1) as p:
        res = dict(p.map(_sweep, SIGMAS + [None]))
    for k, v in res.items():
        print("baseline" if k is None else f"sig_max {k}", v, flush=True)
    json.dump({("baseline" if k is None else str(k)): v for k, v in res.items()},
              open(ROOT / "results" / "sigmax_sweep_oracle_train.json", "w"), indent=1)
