"""Pull IMU / joint / foot-contact arrays out of a Co-RaL ROS1 bag into a compact .npz.

usage: python extract.py <path/to/sequence.bag> <out.npz>
Needs: pip install rosbags   (no ROS install required)
Layout written (rows sorted later by common.load_raw):
  imu    (N, 12): [header_t, record_t, gyro xyz, acc xyz, quat xyzw]
  joints (N, 38): [header_t, record_t, pos x12, vel x12, effort x12]
  feet   (N, 18): [t, record_t, (contact, px, py, pz) x 4 feet FL FR RL RR]
The feet message has no header, so its time is the bag record time.
"""
import sys, numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader


def hs(m, rec):
    h = getattr(m, 'header', None)
    if h is not None and (h.stamp.sec or h.stamp.nanosec):
        return h.stamp.sec + h.stamp.nanosec * 1e-9
    return rec


bag, out = Path(sys.argv[1]), sys.argv[2]
imu = []; js = []; ft = []
with AnyReader([bag]) as r:
    cons = [c for c in r.connections if c.topic in ("/imu", "/joint_states", "/spot/status/feet")]
    for c, t, raw in r.messages(connections=cons):
        m = r.deserialize(raw, c.msgtype); rec = t * 1e-9
        if c.topic == "/imu":
            imu.append([hs(m, rec), rec, m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z,
                        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z,
                        m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w])
        elif c.topic == "/joint_states":
            js.append([hs(m, rec), rec, *m.position, *m.velocity, *m.effort])
        else:
            row = [hs(m, rec), rec]
            for s in m.states[:4]:
                row += [s.contact, s.foot_position_rt_body.x, s.foot_position_rt_body.y, s.foot_position_rt_body.z]
            ft.append(row)
np.savez_compressed(out, imu=np.array(imu), joints=np.array(js), feet=np.array(ft))
print(out, len(imu), len(js), len(ft))
