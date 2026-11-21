"""Choose the continuous-replay settings of the baseline EKF on the TRAINING sequences only.

Grid: update rate x foot measurement noise x foothold random walk (fully trusted foot).
Criterion: mean RPE over the 4 training sequences x 5 timing offsets (same criterion used to pick sigma_max).
Writes results/ekf_tuning.json; run_ekf.py reads the chosen values from there.
usage: python tune_ekf.py [n_workers]
"""
import numpy as np, json, sys, itertools
from multiprocessing import Pool
from common import *
from experiment import OFFSETS

GRID = dict(UPDATE_HZ=[10.0, 25.0, 50.0, 100.0], SIGMA_MEAS=[0.005, 0.01, 0.02, 0.04], SIG_MIN=[1e-4, 1e-3, 1e-2])


def job(args):
    cfg, seq, off = args
    import run_ekf
    from metrics import evaluate
    run_ekf.configure(UPDATE_MODE="continuous", **cfg)
    run_ekf.FOOT_DT = off
    T, P, Q = run_ekf.run(seq, fix_timing=True)
    r = evaluate(seq, T, P, Q)
    return dict(cfg=cfg, seq=seq, offset=off, **{k: float(r[k]) for k in ["ATE", "RPE", "drift"]})


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    cfgs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    jobs = [(c, s, o) for c in cfgs for s in TRAIN for o in OFFSETS]
    with Pool(nw) as p:
        rows = p.map(job, jobs, chunksize=1)
    table = []
    for c in cfgs:
        sel = [r for r in rows if r["cfg"] == c]
        table.append(dict(**c, **{k: float(np.mean([r[k] for r in sel])) for k in ["ATE", "RPE", "drift"]},
                          ATE_std_over_offsets=float(np.mean([np.std([r["ATE"] for r in sel if r["seq"] == s]) for s in TRAIN]))))
    table.sort(key=lambda r: r["RPE"])
    best = {k: table[0][k] for k in GRID}
    for r in table[:12]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    json.dump(dict(criterion="mean RPE over TRAIN x offsets", chosen=best, table=table, rows=rows),
              open(ROOT / "results" / "ekf_tuning.json", "w"), indent=1)
    print("chosen", best)
