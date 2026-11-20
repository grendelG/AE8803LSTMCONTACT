"""Experiment B (proposal fallback): synthetic slips with exactly-known labels.

For each stance phase of each foot, with probability P_SLIP the foot slides in the world by
D ~ U(3, 10) cm BACKWARD (opposite the robot's heading, +-30 deg, plus up to 0.3 sink) over tau ~ U(0.05, 0.15) s
-- a push-off slip, which biases leg odometry in a consistent direction (v1 used random directions,
which mostly cancelled and left the baseline EKF unaffected),
starting at a random point in the stance, and stays displaced until liftoff.
What the robot measures (foot position in body frame) is corrupted by R_wb^T * delta(t);
the EKF, which assumes a planted foot is still, gets dragged by the slip.

Labels: s = exp(-(0.1 s * |d delta/dt|)^2 / (2 * 0.02^2)) for stance feet (slip speed within the stance)
(same functional form and sigma as the LIO labels, but noise-free).
Joint angle/velocity/torque features are zeroed in this experiment: the injection only
corrupts the reported foot position, so they would leak the answer.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from common import *
from features import FEATURE_NAMES, SIGMA_LABEL, DIFF_LAG, dedupe

P_SLIP = 0.2
JOINT_COLS = [FEATURE_NAMES.index(n) for n in FEATURE_NAMES if n.startswith(("q0", "q1", "q2", "dq", "tau"))]


def inject(d, seed):
    """Return a copy of d with corrupted pfoot, plus the world-frame slip delta (N_ft,4,3)."""
    rng = np.random.default_rng(seed)
    t, keep = d["t_ft"], None
    c = d["contact_raw"] == 1
    Rwb = Rot.from_quat(zoh(d["t_imu"], d["q_imu"], t))
    delta = np.zeros(d["pfoot"].shape)
    events = []
    for i in range(4):
        k = 0; N = len(t)
        while k < N:
            if not c[k, i]:
                k += 1; continue
            k0 = k
            while k < N and c[k, i]:
                k += 1
            k1 = k                     # stance = [k0, k1)
            dur = t[k1 - 1] - t[k0]
            if dur > 0.15 and rng.random() < P_SLIP:
                D = rng.uniform(0.03, 0.10); tau = rng.uniform(0.05, 0.15)
                yaw = Rwb[k0].as_euler("zyx")[0]
                th = yaw + np.pi + rng.uniform(-np.pi / 6, np.pi / 6); sink = rng.uniform(0, 0.3)
                dirv = np.array([np.cos(th), np.sin(th), -sink]); dirv /= np.linalg.norm(dirv)
                ts = t[k0] + rng.uniform(0.1, 0.6) * dur
                tt = t[k0:k1]
                u = np.clip((tt - ts) / tau, 0, 1); u = u * u * (3 - 2 * u)   # smoothstep
                delta[k0:k1, i] = (D * u)[:, None] * dirv[None, :]
                events.append((i, ts, D))
    out = dict(d)
    pb = d["pfoot"].copy()
    for i in range(4):
        pb[:, i] += Rwb.inv().apply(delta[:, i])
    out["pfoot"] = pb
    return out, delta, events


def build_synth(seq, seed, foot_dt=0.0):
    """Features (same definition as features.py, joints zeroed) + exact labels for an injected copy."""
    import features
    d0 = load_raw(seq)
    d, delta, ev = inject(d0, seed)
    orig = features.load_raw
    features.load_raw = lambda s: d                # reuse the feature builder on the corrupted data
    old_dt = features.FOOT_DT; features.FOOT_DT = foot_dt
    try:
        r = features.build(seq)
    finally:
        features.load_raw = orig; features.FOOT_DT = old_dt
    X = r["X"]; X[..., JOINT_COLS] = 0.0
    t = r["t"]
    tf = d["t_ft"] + foot_dt
    # slip speed computed only inside each stance (delta resets at liftoff, which is not a slip)
    tf_u, dl, cu = dedupe(tf, delta, d["contact_raw"] == 1)
    v = np.zeros(dl.shape)
    dt_ = np.diff(tf_u)[:, None, None]
    fwd = (dl[1:] - dl[:-1]) / dt_
    same = (cu[1:] & cu[:-1])[..., None]
    fwd = np.where(same, fwd, 0.0)
    v[1:] += 0.5 * fwd; v[:-1] += 0.5 * fwd
    dd = np.linalg.norm(interp(tf_u, v, t), axis=2) * 0.1      # displacement per 0.1 s at the current slip speed
    y = np.exp(-dd ** 2 / (2 * SIGMA_LABEL ** 2)).astype(np.float32)
    y[r["contact"] < 0.5] = np.nan
    return dict(t=t, X=X, y=y, contact=r["contact"], pfoot=d["pfoot"], n_events=len(ev))


if __name__ == "__main__":
    import json
    (ROOT / "features_synth").mkdir(exist_ok=True)
    for s in SEQS:
        seed = 1000 + SEQS.index(s)
        r = build_synth(s, seed)
        np.savez_compressed(ROOT / "features_synth" / f"{s}.npz", **{k: v for k, v in r.items()})
        y = r["y"]
        print(f"{s}: injected slips {r['n_events']}, labelled steps {np.sum(~np.isnan(y))}, frac score<0.5 {np.nanmean(y<0.5):.3f}")
