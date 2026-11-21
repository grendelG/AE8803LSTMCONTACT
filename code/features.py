"""Build per-foot, IMU-rate feature tensors and reliability labels for every sequence.

Clock convention (calib.py): the feet/joint streams are shifted by FOOT_DT = calib "feet_dt" (about +5 ms)
onto the IMU clock; LIO poses by calib "time_offset" (+0.06 s). Labels: labels.window_displacements.
"""
import numpy as np, json
from common import *
from labels import window_displacements

CAL = json.load(open(ROOT / "results" / "calib.json"))
FOOT_DT = CAL.get("feet_dt", 0.06)   # feet/joint streams -> IMU clock (calib.py)
SIGMA_LABEL = 0.02      # m per 0.1 s (a planted foot moving 20 cm/s is clearly slipping); same as synth labels
DIFF_LAG = 3            # samples (30 ms) for causal finite differences

FEATURE_NAMES = (["contact", "t_since_td", "n_stance"] + [f"p_{a}" for a in "xyz"] + [f"pdot_{a}" for a in "xyz"]
                 + [f"vimpl_{a}" for a in "xyz"] + [f"dev_{a}" for a in "xyz"] + ["dev_norm"]
                 + [f"q{j}" for j in range(3)] + [f"dq{j}" for j in range(3)] + [f"tau{j}" for j in range(3)]
                 + [f"gyro_{a}" for a in "xyz"] + [f"acc_{a}" for a in "xyz"] + [f"leg{i}" for i in range(4)])


def dedupe(t, *xs):
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    return (t[keep],) + tuple(x[keep] for x in xs)


def build(seq):
    d = load_raw(seq)
    t = d["t_imu"]; N = len(t)
    tf, pfoot, craw = dedupe(d["t_ft"] + FOOT_DT, d["pfoot"], d["contact_raw"])
    tj, q, dq, tau = dedupe(d["t_js"] + FOOT_DT, d["q"], d["dq"], d["tau"])

    contact = zoh(tf, (craw == 1).astype(float), t)            # causal
    p = interp(tf, pfoot, t)                                    # (N,4,3)
    # causal derivative on the IMU grid
    pdot = np.zeros_like(p)
    pdot[DIFF_LAG:] = (p[DIFF_LAG:] - p[:-DIFF_LAG]) / (t[DIFF_LAG:] - t[:-DIFF_LAG])[:, None, None]
    w = d["gyro"]
    vimpl = -(pdot + np.cross(w[:, None, :], p))                # body vel implied by each foot (if planted)
    st = contact > 0.5
    vm = np.where(st[..., None], vimpl, np.nan)
    with np.errstate(all="ignore"):
        cons = np.nanmedian(vm, axis=1)                         # consensus over stance feet
    cons = np.nan_to_num(cons)
    dev = vimpl - cons[:, None, :]
    dev[~st] = 0.0
    nst = st.sum(1).astype(float)

    tsd = np.zeros((N, 4)); last = np.full(4, t[0] - 1.0)
    for k in range(N):
        for i in range(4):
            if st[k, i] and (k == 0 or not st[k - 1, i]):
                last[i] = t[k]
            tsd[k, i] = min(t[k] - last[i], 1.0) if st[k, i] else 0.0

    qj = interp(tj, q, t).reshape(N, 4, 3)
    dqj = interp(tj, dq, t).reshape(N, 4, 3)
    tauj = interp(tj, tau, t).reshape(N, 4, 3)

    X = np.zeros((N, 4, len(FEATURE_NAMES)), dtype=np.float32)
    for i in range(4):
        X[:, i] = np.column_stack([
            contact[:, i], tsd[:, i], nst, p[:, i], pdot[:, i], vimpl[:, i], dev[:, i],
            np.linalg.norm(dev[:, i], axis=1), qj[:, i], dqj[:, i], tauj[:, i], w, d["acc"],
            np.tile(np.eye(4)[i], (N, 1))])

    # labels on the IMU grid: foot displacement over a centred 0.1 s window, fused LIO+IMU body motion
    _, D = window_displacements(d, CAL, "fused", foot_dt=FOOT_DT)
    disp = np.linalg.norm(D, axis=2)
    disp[~st] = np.nan
    y = np.exp(-disp ** 2 / (2 * SIGMA_LABEL ** 2))
    return dict(t=t, X=X, y=y.astype(np.float32), disp=disp, contact=contact, pfoot_imu=p,
                gyro=w, acc=d["acc"], tf=tf, pfoot=pfoot, craw=craw)


if __name__ == "__main__":
    (ROOT / "features").mkdir(exist_ok=True)
    for s in SEQS:
        r = build(s)
        np.savez_compressed(ROOT / "features" / f"{s}.npz", **r)
        y = r["y"]
        print(f"{s}: N={len(r['t'])}, labelled foot-steps={np.sum(~np.isnan(y))}, mean score {np.nanmean(y):.3f}, "
              f"frac<0.5 {np.nanmean(y<0.5):.3f}")
