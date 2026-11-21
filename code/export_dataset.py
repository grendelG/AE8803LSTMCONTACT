"""Write the team-standard per-sequence files: dataset/<seq>.npz

All arrays share one time base: the IMU timestamps (100 Hz), N rows.
Keys / shapes:
  timestamps          (N,)    s, IMU clock (ROS time)
  imu                 (N, 6)  [gyro_x, gyro_y, gyro_z (rad/s), acc_x, acc_y, acc_z (m/s^2)], body frame
  joint_pos           (N, 12) rad, order FL(hip_x,hip_y,knee), FR, RL, RR
  joint_vel           (N, 12) rad/s, same order
  contact_binary      (N, 4)  1 = stance, 0 = swing; foot order FL, FR, RL, RR
  foot_pos_body       (N, 4, 3) m, Spot-reported foot position in body frame (extra)
  lio_pose            (N, 7)  [x, y, z, qx, qy, qz, qw] LIO reference, already in Spot's body frame
  reliability_scores  (N, 4)  float in [0, 1]; 1.0 wherever there is no label (swing / partial stance)
  label_mask          (N, 4)  bool, True where reliability_scores is a real label (use for training loss)
Feet / joint streams were shifted onto the IMU clock by calib "feet_dt" (about +5 ms) and LIO poses by calib
"time_offset" (+0.06 s), see results/calib.json; then interpolated (zero-order hold for contact).
reliability_scores: foot displacement over a centred 0.1 s window, body motion from the fused LIO+IMU
reference (code/reference.py), score = exp(-d^2 / (2 * 0.02^2)).
"""
import numpy as np
from scipy.spatial.transform import Slerp
from common import *
from features import FOOT_DT, dedupe
from metrics import gt_body

if __name__ == "__main__":
    out = ROOT / "dataset"; out.mkdir(exist_ok=True)
    for s in SEQS:
        d = load_raw(s)
        f = np.load(ROOT / "features" / f"{s}.npz")
        t = d["t_imu"]
        tj, q, dq = dedupe(d["t_js"] + FOOT_DT, d["q"], d["dq"])
        tg, pg, Rg = gt_body(s)
        tc = np.clip(t, tg[0], tg[-1])
        lio = np.column_stack([interp(tg, pg, tc), Slerp(tg, Rg)(tc).as_quat()])
        y = f["y"]; mask = ~np.isnan(y)
        rec = dict(
            timestamps=t.astype(np.float64),
            imu=np.column_stack([d["gyro"], d["acc"]]).astype(np.float32),
            joint_pos=interp(tj, q, t).astype(np.float32),
            joint_vel=interp(tj, dq, t).astype(np.float32),
            contact_binary=(f["contact"] > 0.5).astype(np.uint8),
            foot_pos_body=f["pfoot_imu"].astype(np.float32),
            lio_pose=lio.astype(np.float64),
            reliability_scores=np.where(mask, y, 1.0).astype(np.float32),
            label_mask=mask,
        )
        N = len(t)
        assert rec["imu"].shape == (N, 6) and rec["joint_pos"].shape == (N, 12) and rec["lio_pose"].shape == (N, 7)
        assert rec["reliability_scores"].shape == (N, 4) and rec["contact_binary"].shape == (N, 4)
        assert np.all((rec["reliability_scores"] >= 0) & (rec["reliability_scores"] <= 1))
        np.savez_compressed(out / f"{s}.npz", **rec)
        print(s, {k: v.shape for k, v in rec.items()})
