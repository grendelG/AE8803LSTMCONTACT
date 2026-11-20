"""ATE / RPE / drift against the Co-RaL LIO reference (converted to Spot's body frame)."""
import numpy as np, json
from scipy.spatial.transform import Rotation as Rot, Slerp
from common import *

CAL = json.load(open(ROOT / "results" / "calib.json"))


def gt_body(seq):
    """Reference body trajectory on the IMU clock: times, positions, Rotations."""
    d = load_raw(seq)
    R_sb, t_sb = np.array(CAL["R_sb"]), np.array(CAL["t_sb"])
    Rws = gt_rot(d)
    p = d["p_gt"] + Rws.apply(t_sb)
    R = Rws * Rot.from_matrix(R_sb)
    return d["t_gt"] + CAL["time_offset"], p, R


def umeyama(A, B):
    """R,t minimizing ||R A + t - B|| (no scale)."""
    ma, mb = A.mean(0), B.mean(0)
    H = (A - ma).T @ (B - mb)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mb - R @ ma


def evaluate(seq, T, P, Q, rpe_dist=10.0, drift_align_s=10.0):
    tg, pg, Rg = gt_body(seq)
    m = (tg >= T[0]) & (tg <= T[-1])
    tg, pg, Rg = tg[m], pg[m], Rg[m]
    pe = interp(T, P, tg)
    Re = Slerp(T, Rot.from_quat(Q))(tg)
    # ATE (SE3 alignment over the whole trajectory)
    R, t = umeyama(pe, pg)
    ate = np.sqrt(np.mean(np.sum((pe @ R.T + t - pg) ** 2, axis=1)))
    # RPE: relative translation error over ~rpe_dist metres of travelled path
    seg = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pg, axis=0), axis=1))])
    errs = []
    for i in range(len(tg)):
        j = np.searchsorted(seg, seg[i] + rpe_dist)
        if j >= len(tg):
            break
        dg = Rg[i].inv().apply(pg[j] - pg[i])
        de = Re[i].inv().apply(pe[j] - pe[i])
        errs.append(np.linalg.norm(dg - de))
    rpe = np.sqrt(np.mean(np.square(errs)))
    # drift: align on the first drift_align_s seconds, final error / path length
    a = tg <= tg[0] + drift_align_s
    R2, t2 = umeyama(pe[a], pg[a])
    final = np.linalg.norm(R2 @ pe[-1] + t2 - pg[-1])
    drift = 100 * final / seg[-1]
    return dict(ATE=ate, RPE=rpe, drift=drift, final_err=final, path=seg[-1])


if __name__ == "__main__":
    from run_ekf import run
    for s in SEQS:
        for fix in [False, True]:
            T, P, Q = run(s, fix_timing=fix)
            r = evaluate(s, T, P, Q)
            print(s, "fix_timing" if fix else "orig      ", {k: round(v, 3) for k, v in r.items()})
