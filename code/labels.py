"""Per-foot contact-reliability labels from the LIO reference.

For each 0.1 s reference interval and each foot in contact for the whole interval:
  d = || f_world(t_{k+1}) - f_world(t_k) ||,   f_world = T_world_sensor * T_sensor_body * p_foot_body
A planted foot should have d ~ 0. Score s = exp(-d^2 / (2 sigma^2)).
"""
import numpy as np, json
from common import *

CAL = json.load(open(ROOT / "results" / "calib.json")) if (ROOT / "results" / "calib.json").exists() else None


def stance_displacements(d, cal, foot_dt=0.0):
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


if __name__ == "__main__":
    d = load_raw("building")
    for fdt in [-0.06, -0.03, 0.0, 0.03, 0.06]:
        _, _, D = stance_displacements(d, CAL, fdt)
        print(f"foot dt {fdt:+.2f}: median {np.nanmedian(D)*100:.2f} cm, 90% {np.nanpercentile(D,90)*100:.2f} cm")
    for tz in [None, -0.15, 0.0]:
        cal = dict(CAL)
        if tz is not None:
            cal["t_sb"] = [CAL["t_sb"][0], CAL["t_sb"][1], tz]
        _, _, D = stance_displacements(d, cal)
        print("t_z", tz, f"median {np.nanmedian(D)*100:.2f} cm  p90 {np.nanpercentile(D,90)*100:.2f} cm")
    cal0 = dict(CAL); cal0["t_sb"] = [0, 0, 0]
    _, _, D = stance_displacements(d, cal0)
    print("t=0", f"median {np.nanmedian(D)*100:.2f} cm  p90 {np.nanpercentile(D,90)*100:.2f}")
