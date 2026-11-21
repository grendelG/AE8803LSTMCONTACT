"""Fused body-motion reference for labelling: LIO for slow motion, IMU for fast motion.

Why: LIO (10 Hz, one pose per LiDAR sweep) is accurate over metres and seconds but jitters at the gait
frequency (Spot bounces ~3.6 times/s). Over 0.1 s, which is what a slip label needs, that jitter is as big
as the label itself. The IMU is the opposite: smooth over 0.1 s, drifts over seconds. A complementary
filter takes each sensor where it is good:

  orientation : gyro integration (bias tracked from LIO, slowly) + slow pull toward LIO attitude
  velocity    : LP(v_LIO) + HP(v_IMU),  v_IMU = integral of (R * f_imu + g)
with LP + HP = identity (same zero-phase Butterworth, crossover FC). Offline / non-causal: used only
to build training labels and the oracle, never as an input to the predictors or the EKF.

Everything is returned on the IMU time grid (IMU clock).
"""
import numpy as np, json
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as Rot, Slerp
from common import *

FC = 1.0            # Hz, LIO/IMU crossover; below Spot's gait band (~1.8 Hz stride, ~3.6 Hz bounce).
                    # Chosen on TRAIN: planted-foot motion falls from 1.82 cm (0.1 Hz) to 0.95 cm (0.8 Hz) and
                    # flattens by 1.2 Hz; above that LIO's gait-band jitter would leak in.
FC_BIAS = 0.02      # Hz, gyro-bias tracking bandwidth
G = 9.81


def _lp(x, fs, fc, order=2):
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, x, axis=0, padtype="odd", padlen=min(len(x) - 1, int(3 * fs / fc)))


def fused_reference(d, cal, fc=FC):
    """d = load_raw(seq). Returns dict(t, R (Rotation, world<-body), v (N,3) world body-origin velocity,
    disp_cum (N,3) cumulative integral of v, i.e. body position up to a constant)."""
    tg, pg, Rg = gt_body_from(d, cal)                # LIO trajectory of the body origin, IMU clock
    t = d["t_imu"]
    m = (t >= tg[0]) & (t <= tg[-1])
    t = t[m]; gyro = d["gyro"][m]; acc = d["acc"][m]
    # uniform grid for filtering
    fs = 100.0
    tu = np.arange(t[0], t[-1], 1 / fs)
    w = interp(t, gyro, tu); f = interp(t, acc, tu)

    # --- orientation ---------------------------------------------------------
    # gyro bias = (gyro - LIO angular rate), averaged per LIO interval, then low-passed hard
    dtg = np.diff(tg)
    w_lio = (Rg[:-1].inv() * Rg[1:]).as_rotvec() / dtg[:, None]
    C = np.vstack([np.zeros((1, 3)), np.cumsum(w[:-1] / fs, axis=0)])
    tmid = 0.5 * (tg[1:] + tg[:-1])
    ok = (tg[:-1] >= tu[0]) & (tg[1:] <= tu[-1])
    w_imu = (interp(tu, C, tg[1:][ok]) - interp(tu, C, tg[:-1][ok])) / dtg[ok, None]
    b10 = _lp(w_imu - w_lio[ok], 10.0, FC_BIAS)
    bias = interp(tmid[ok], b10, tu)
    # integrate bias-corrected gyro
    dR = Rot.from_rotvec((w - bias) / fs)
    Rs = [Slerp(tg, Rg)([tu[0]])[0]]
    for k in range(len(tu) - 1):
        Rs.append(Rs[-1] * dR[k])
    Rgyr = Rot.concatenate(Rs)
    # slow correction toward LIO attitude (world-side), same crossover as velocity
    Rl = Slerp(tg, Rg)(np.clip(tu, tg[0], tg[-1]))
    e = (Rl * Rgyr.inv()).as_rotvec()
    R = Rot.from_rotvec(_lp(e, fs, fc)) * Rgyr

    # --- velocity --------------------------------------------------------------
    a_w = R.apply(f) + np.array([0, 0, -G])
    v_imu = np.cumsum(a_w, axis=0) / fs
    v_imu -= _lp(v_imu, fs, fc)                         # HP part
    # LIO velocity: central differences of positions, onto the grid, LP part
    v_l10 = np.gradient(pg, tg, axis=0)
    v_lio = _lp(interp(tg, v_l10, tu), fs, fc)
    v = v_lio + v_imu
    disp = np.vstack([np.zeros((1, 3)), np.cumsum(0.5 * (v[1:] + v[:-1]) / fs, axis=0)])

    # back onto the IMU timestamps
    out_t = d["t_imu"]
    tc = np.clip(out_t, tu[0], tu[-1])
    return dict(t=out_t, R=Slerp(tu, R)(tc), v=interp(tu, v, tc), disp_cum=interp(tu, disp, tc),
                valid=(out_t >= tu[0]) & (out_t <= tu[-1]))


def gt_body_from(d, cal):
    """Same as metrics.gt_body but from an already-loaded sequence dict."""
    R_sb, t_sb = np.array(cal["R_sb"]), np.array(cal["t_sb"])
    Rws = gt_rot(d)
    return d["t_gt"] + cal["time_offset"], d["p_gt"] + Rws.apply(t_sb), Rws * Rot.from_matrix(R_sb)
