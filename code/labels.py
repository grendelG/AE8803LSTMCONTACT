"""Per-foot contact-reliability labels.

Current labels (window_displacements, used by features.py):
  for every IMU step t and every foot in contact for the whole window [t - 0.05 s, t + 0.05 s]:
    f_world(t) = p_body(t) + R_body(t) * p_foot_body(t)       (body motion from the fused LIO+IMU reference)
    d          = || f_world(t + 0.05) - f_world(t - 0.05) ||   (a planted foot should not move)
    score      = exp(-d^2 / (2 sigma^2)),  sigma = 2 cm
  Feet stream on the IMU clock with the calibrated shift calib.json["feet_dt"].

Legacy labels (stance_displacements, version 1 of the project): same idea, but body motion straight from the
10 Hz LIO poses, one label per LIO interval, feet shifted by the LIO time offset (+0.06 s). Those labels were
~97% shared between the two planted feet (reference + timing error, not slip); see README section 5.
"""
import numpy as np, json
from common import *

CAL = json.load(open(ROOT / "results" / "calib.json")) if (ROOT / "results" / "calib.json").exists() else None


def stance_displacements(d, cal, foot_dt=0.0):
    """Legacy (v1) labels: displacement over each 10 Hz LIO interval, LIO-only body motion."""
    R_sb, t_sb, off = np.array(cal["R_sb"]), np.array(cal["t_sb"]), cal["time_offset"]
    tg = d["t_gt"] + off                    # reference times on the IMU clock
    R = gt_rot(d).as_matrix(); p = d["p_gt"]
    tf = d["t_ft"] + foot_dt
    pf = interp(tf, d["pfoot"], tg)
    c = d["contact_raw"] == 1
    # contact for the whole interval: count of non-contact samples in each interval
    idx = np.searchsorted(tf, tg)
    nc = np.vstack([np.zeros((1, 4)), np.cumsum(~c, axis=0)])
    bad = nc[np.clip(idx[1:], 0, len(tf))] - nc[np.clip(idx[:-1], 0, len(tf))]
    # also require contact at the sample just before each boundary
    c_at = zoh(tf, c, tg)
    full = (bad == 0) & c_at[:-1] & c_at[1:]
    fw = p[:, None, :] + np.einsum("kij,knj->kni", R, (pf @ R_sb.T) + t_sb)
    disp = np.linalg.norm(fw[1:] - fw[:-1], axis=2)          # (K-1, 4)
    disp[~full] = np.nan
    return tg[:-1], tg[1:], disp


def foot_world_tracks(d, cal, ref="fused", foot_dt=None):
    """World position of every foot on the IMU grid, f = p_body + R_body * p_foot_body (positions up to a
    constant per sequence, which cancels in displacements). ref = "fused" (reference.py) or "lio"."""
    from features import dedupe
    if foot_dt is None:
        foot_dt = cal.get("feet_dt", 0.0)
    t = d["t_imu"]
    tf, pfoot, craw = dedupe(d["t_ft"] + foot_dt, d["pfoot"], d["contact_raw"])
    pf = interp(tf, pfoot, t)
    if ref == "fused":
        from reference import fused_reference
        r = fused_reference(d, cal)
        p, R, valid = r["disp_cum"], r["R"], r["valid"]
    else:
        from reference import gt_body_from
        from scipy.spatial.transform import Slerp
        tg, pg, Rg = gt_body_from(d, cal)
        tc = np.clip(t, tg[0], tg[-1])
        p, R, valid = interp(tg, pg, tc), Slerp(tg, Rg)(tc), (t >= tg[0]) & (t <= tg[-1])
    Rm = R.as_matrix()
    fw = p[:, None, :] + np.einsum("kij,knj->kni", Rm, pf)
    return t, fw, tf, craw == 1, valid


def window_displacements(d, cal, ref="fused", win=0.1, foot_dt=None):
    """Per IMU step and foot: world displacement vector of the foot over [t - win/2, t + win/2].
    NaN unless the foot is in contact for the whole window (feet-stream samples) and the reference is valid.
    Returns (t, D (N,4,3))."""
    t, fw, tf, c, valid = foot_world_tracks(d, cal, ref, foot_dt)
    ta, tb = t - win / 2, t + win / 2
    D = interp(t, fw, np.clip(tb, t[0], t[-1])) - interp(t, fw, np.clip(ta, t[0], t[-1]))
    nc = np.vstack([np.zeros((1, 4)), np.cumsum(~c, axis=0)])
    ia, ib = np.searchsorted(tf, ta), np.searchsorted(tf, tb, side="right")
    bad = nc[np.clip(ib, 0, len(tf))] - nc[np.clip(ia, 0, len(tf))]
    full = (bad == 0) & zoh(tf, c, ta) & zoh(tf, c, tb)
    full &= (valid & (ta >= t[0]) & (tb <= t[-1]))[:, None]
    D[~full] = np.nan
    return t, D


if __name__ == "__main__":
    # quick look: legacy (LIO, 10 Hz intervals) vs current (fused, sliding 0.1 s window) label noise floor
    for seq in SEQS:
        d = load_raw(seq)
        _, _, D0 = stance_displacements(d, CAL, 0.06)
        _, D1 = window_displacements(d, CAL, "fused")
        n1 = np.linalg.norm(D1, axis=2)
        print(f"{seq:8s} legacy median {np.nanmedian(D0)*100:.2f} cm | current median {np.nanmedian(n1)*100:.2f} cm, "
              f"p99 {np.nanpercentile(n1, 99)*100:.2f} cm")
