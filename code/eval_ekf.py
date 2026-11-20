"""Run every method x sig_max x sequence x timing-offset through the EKF; save raw metrics as JSON.
usage: python eval_ekf.py <experiment: real|synth> <method> [<method> ...]
"""
import numpy as np, json, sys
import run_ekf
from metrics import evaluate
from common import *
from experiment import OFFSETS

SIGMAS = [1e-2, 3e-2, 1e-1]


def load_override(exp, seq):
    if exp == "real":
        return None
    from synth import inject
    return inject(load_raw(seq), 1000 + SEQS.index(seq))[0]


if __name__ == "__main__":
    exp, methods = sys.argv[1], sys.argv[2:]
    sdir = ROOT / "results" / ("scores" if exp == "real" else "scores_synth")
    out = ROOT / "results" / f"ekf_{exp}"; out.mkdir(exist_ok=True, parents=True); (out / "traj").mkdir(exist_ok=True)
    sigmas = SIGMAS if exp == "real" else [3e-2, 1e-1, 3e-1, 1.0]
    for m in methods:
        res = []
        for s in SEQS:
            dov = load_override(exp, s)
            sc = None if m == "baseline" else np.load(sdir / f"{m}_{s}.npy")
            for sm in ([None] if m == "baseline" else sigmas):
                for off in OFFSETS:
                    run_ekf.FOOT_DT = off
                    T, P, Q = run_ekf.run(s, scores=sc, sig_max=sm, fix_timing=True, d_override=dov)
                    r = evaluate(s, T, P, Q)
                    np.savez_compressed(out / "traj" / f"{m}_{s}_{sm}_{off:+.2f}.npz", T=T[::10], P=P[::10])
                    r.update(seq=s, method=m, sig_max=sm, offset=off)
                    res.append({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()})
            print(m, s, "done", flush=True)
        json.dump(res, open(out / f"{m}.json", "w"), indent=0)
