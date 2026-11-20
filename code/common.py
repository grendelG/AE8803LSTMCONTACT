"""Shared data loading, geometry, and frame calibration for the final project."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot, Slerp

ROOT = Path(__file__).resolve().parent.parent   # the "Final Project" folder
SEQS = ["building", "under", "garage", "narrow", "stair", "trail"]
TRAIN = ["building", "under", "garage", "narrow"]
TEST = ["stair", "trail"]
FEET = ["FL", "FR", "RL", "RR"]


def load_raw(seq):
    d = np.load(ROOT / "extracted" / f"{seq}.npz")
    imu, js, ft = d["imu"], d["joints"], d["feet"]
    imu = imu[np.argsort(imu[:, 0])]
    js = js[np.argsort(js[:, 0])]
    ft = ft[np.argsort(ft[:, 0])]
    out = dict(
        t_imu=imu[:, 0], gyro=imu[:, 2:5], acc=imu[:, 5:8], q_imu=imu[:, 8:12],
        t_js=js[:, 0], q=js[:, 2:14], dq=js[:, 14:26], tau=js[:, 26:38],
        t_ft=ft[:, 0],
        contact_raw=ft[:, 2::4][:, :4].astype(int),
        pfoot=np.stack([ft[:, 3 + 4 * i: 6 + 4 * i] for i in range(4)], axis=1),  # (N,4,3)
    )
    g = np.loadtxt(ROOT / "gt" / f"{seq}.txt")
    out["t_gt"], out["p_gt"], out["q_gt"] = g[:, 0], g[:, 1:4], g[:, 4:8]
    return out


def interp(t_src, x_src, t_dst):
    x_src = np.asarray(x_src)
    flat = x_src.reshape(len(t_src), -1)
    res = np.stack([np.interp(t_dst, t_src, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return res.reshape((len(t_dst),) + x_src.shape[1:])


def zoh(t_src, x_src, t_dst):
    idx = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, len(t_src) - 1)
    return x_src[idx]


def gt_rot(seq_d):
    return Rot.from_quat(seq_d["q_gt"])


def gt_pose_at(seq_d, t):
    """Interpolated GT (sensor-frame) pose at times t: R (Rotation), p (N,3)."""
    tg = seq_d["t_gt"]
    t = np.clip(t, tg[0], tg[-1])
    sl = Slerp(tg, gt_rot(seq_d))
    return sl(t), interp(tg, seq_d["p_gt"], t)
