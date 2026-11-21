"""Run every method x sig_max x sequence x timing-offset through the EKF; save raw metrics as JSON.
usage: python eval_ekf.py <experiment: real|synth> <method> [<method> ...] [-j N]

Methods: baseline     continuous-update EKF, fixed noise (the tuned replay, run_ekf.py)
         baseline_td  original Project 2b replay (updates only at touchdowns), fixed noise
         heuristic | mlp | lstm | oracle   continuous-update EKF with per-foot reliability scores
"""
import numpy as np, json, sys
from multiprocessing import Pool
from common import *
from experiment import OFFSETS, SIGMAS


def load_override(exp, seq):
    if exp == "real":
        return None
    from synth import inject
    return inject(load_raw(seq), 1000 + SEQS.index(seq))[0]


def job(a):
    exp, m, s, sm, off = a
    import run_ekf
    from metrics import evaluate
    # pool workers are reused across jobs: always set the filter configuration explicitly
    run_ekf.configure(**(run_ekf.TOUCHDOWN_CFG if m == "baseline_td" else run_ekf.CONTINUOUS_CFG))
    sdir = ROOT / "results" / ("scores" if exp == "real" else "scores_synth")
    sc = None if m.startswith("baseline") else np.load(sdir / f"{m}_{s}.npy")
    run_ekf.FOOT_DT = off
    T, P, Q = run_ekf.run(s, scores=sc, sig_max=sm, fix_timing=True, d_override=load_override(exp, s))
    r = evaluate(s, T, P, Q)
    out = ROOT / "results" / f"ekf_{exp}" / "traj"
    np.savez_compressed(out / f"{m}_{s}_{sm}_{off:+.2f}.npz", T=T[::10], P=P[::10])
    r.update(seq=s, method=m, sig_max=sm, offset=off)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}


if __name__ == "__main__":
    args = sys.argv[1:]
    nw = 16
    if "-j" in args:
        i = args.index("-j"); nw = int(args[i + 1]); del args[i:i + 2]
    exp, methods = args[0], args[1:]
    out = ROOT / "results" / f"ekf_{exp}"; (out / "traj").mkdir(exist_ok=True, parents=True)
    jobs = [(exp, m, s, sm, off) for m in methods for s in SEQS
            for sm in ([None] if m.startswith("baseline") else SIGMAS) for off in OFFSETS]
    with Pool(nw) as p:
        rows = p.map(job, jobs, chunksize=1)
    for m in methods:
        json.dump([r for r in rows if r["method"] == m], open(out / f"{m}.json", "w"), indent=0)
        print(m, "done", flush=True)
