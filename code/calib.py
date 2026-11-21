"""Estimate the fixed transform between the LIO reference frame and Spot's body frame.

T_world_body(t) = T_world_sensor(t) * T_sensor_body
Rotation from gyro vs LIO angular-rate alignment (Kabsch); translation from
requiring stance feet to stay still in the world (linear least squares).
"""
import numpy as np, json
from scipy.spatial.transform import Rotation as Rot
from common import *


def gt_angvel(d):
    R = gt_rot(d)
    dt = np.diff(d["t_gt"])
    w = (R[:-1].inv() * R[1:]).as_rotvec() / dt[:, None]   # sensor frame
    tm = 0.5 * (d["t_gt"][:-1] + d["t_gt"][1:])
    return tm, w, d["t_gt"][:-1], d["t_gt"][1:]


def imu_avg(d, t0, t1, x):
    T = d["t_imu"]
    C = np.vstack([np.zeros((1, 3)), np.cumsum(0.5 * (x[1:] + x[:-1]) * np.diff(T)[:, None], axis=0)])
    a = interp(T, C, t0); b = interp(T, C, t1)
    return (b - a) / (t1 - t0)[:, None]


def kabsch(A, B):
    """R minimizing ||R A_i - B_i||."""
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def stance_mask(d):
    return d["contact_raw"] == 1


def calibrate(seqs, dt_off=0.0):
    A, B = [], []
    for s in seqs:
        d = load_raw(s)
        tm, w, t0, t1 = gt_angvel(d)
        wi = imu_avg(d, t0 + dt_off, t1 + dt_off, d["gyro"])
        ok = (np.linalg.norm(w, axis=1) > 0.05)
        A.append(w[ok]); B.append(wi[ok])
    A, B = np.vstack(A), np.vstack(B)
    R_bs = kabsch(A, B)                       # body <- sensor
    res = np.linalg.norm(B - A @ R_bs.T, axis=1)
    return R_bs, np.median(res), np.sqrt(np.mean(res ** 2))


def solve_translation(seqs, R_sb):
    """Stance foot world position f = p_ws + R_ws (R_sb p_b + t_sb) constant between GT ticks.
    => [R_ws(k+1) - R_ws(k)] t_sb = -(p(k+1)-p(k)) - [R(k+1)-R(k)] R_sb p_b
    """
    rows, rhs = [], []
    for s in seqs:
        d = load_raw(s)
        tg = d["t_gt"]; R = gt_rot(d).as_matrix(); p = d["p_gt"]
        pf = interp(d["t_ft"], d["pfoot"], tg)
        st = zoh(d["t_ft"], (d["contact_raw"] == 1).astype(float), tg) > 0.5
        for k in range(len(tg) - 1):
            dR = R[k + 1] - R[k]
            for i in range(4):
                if st[k, i] and st[k + 1, i]:
                    # require contact the whole interval
                    m = (d["t_ft"] >= tg[k]) & (d["t_ft"] <= tg[k + 1])
                    if not np.all(d["contact_raw"][m, i] == 1):
                        continue
                    rows.append(dR)
                    rhs.append(-(p[k + 1] - p[k]) - (R[k + 1] @ R_sb @ pf[k + 1, i] - R[k] @ R_sb @ pf[k, i]))
    Aa, bb = np.vstack(rows), np.concatenate(rhs)
    # robust: IRLS with Huber
    t = np.linalg.lstsq(Aa, bb, rcond=None)[0]
    for _ in range(10):
        r = (Aa @ t - bb).reshape(-1, 3)
        n = np.linalg.norm(r, axis=1)
        w = np.repeat(np.minimum(1.0, 0.01 / np.maximum(n, 1e-9)), 3)
        t = np.linalg.lstsq(Aa * w[:, None], bb * w, rcond=None)[0]
    return t


if __name__ == "__main__":
    # time offset scan
    best = None
    for off in np.arange(-0.2, 0.21, 0.02):
        R_bs, med, rms = calibrate(TRAIN, off)
        print(f"off {off:+.2f}  med {med:.4f} rms {rms:.4f}")
        if best is None or rms < best[2]:
            best = (off, R_bs, rms)
    off, R_bs, rms = best
    print("best offset", off, "\nR_bs\n", R_bs.round(4), "\nrpy deg", Rot.from_matrix(R_bs).as_euler("xyz", degrees=True).round(2))
    # check on test seqs too (not used for fitting)
    for s in TEST:
        print(s, "test-check", calibrate([s], off)[1:])
    R_sb = R_bs.T
    t_sb = solve_translation(TRAIN, R_sb)
    print("t_sb (body origin in sensor frame)", t_sb.round(4))
    cal = dict(time_offset=float(off), R_sb=R_sb.tolist(), t_sb=t_sb.tolist())

    # Feet/joint stream vs IMU clock. Shift that makes planted feet move least against the fused
    # LIO+IMU reference (reference.py). The LIO-only reference cannot resolve this: at 10 Hz it smears the
    # gait-band motion, which is exactly what a feet-timing error shows up in.
    from labels import window_displacements
    raw = {s: load_raw(s) for s in TRAIN}
    scan = {}
    for fdt in np.round(np.arange(-0.03, 0.0451, 0.005), 3):
        scan[float(fdt)] = float(np.mean([np.nanmedian(np.linalg.norm(
            window_displacements(raw[s], cal, "fused", foot_dt=fdt)[1], axis=2)) for s in TRAIN]))
        print(f"feet shift {fdt:+.3f} s: median planted-foot motion {scan[float(fdt)]*100:.3f} cm / 0.1 s")
    cal["feet_dt"] = min(scan, key=scan.get)
    cal["feet_dt_scan"] = scan
    print("feet/joint -> IMU clock shift", cal["feet_dt"])
    json.dump(cal, open(ROOT / "results" / "calib.json", "w"), indent=1)
