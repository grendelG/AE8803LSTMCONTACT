# Dataset files: one `.npz` per Co-RaL sequence

Sequences: `building`, `under`, `garage`, `narrow` (train) · `stair`, `trail` (test)

All arrays share **one time base: the IMU timestamps (100 Hz), N rows.**

| Key | Shape | Contents |
|---|---|---|
| `timestamps` | `(N,)` | seconds, IMU clock (ROS time) |
| `imu` | `(N, 6)` | `[gyro_x, gyro_y, gyro_z]` rad/s, `[acc_x, acc_y, acc_z]` m/s², body frame |
| `joint_pos` | `(N, 12)` | rad; order FL(hip_x, hip_y, knee), FR, RL, RR |
| `joint_vel` | `(N, 12)` | rad/s, same order |
| `contact_binary` | `(N, 4)` | 1 = stance, 0 = swing; feet FL, FR, RL, RR |
| `lio_pose` | `(N, 7)` | `[x, y, z, qx, qy, qz, qw]` LIO reference, **already converted to Spot's body frame** |
| `reliability_scores` | `(N, 4)` | float in [0, 1]; set to 1.0 wherever there is no label |
| `label_mask` | `(N, 4)` | bool; True where `reliability_scores` is a real label. **Mask your loss with this.** |
| `foot_pos_body` | `(N, 4, 3)` | extra: Spot-reported foot position in body frame, m |

```python
d = np.load("dataset/stair.npz")
y, m = d["reliability_scores"], d["label_mask"]
loss = ((pred - y)**2)[m].mean()
```

## How the labels are made
For every 0.1 s LIO interval and every foot in contact for the whole interval, the foot is placed in the world
(LIO pose × body→foot) and its displacement `d` is measured. A planted foot should not move:
`score = exp(-d² / (2·0.02²))`. Swing feet and partial-stance intervals get no label.

## Timing / frames (see `results/calib.json`)
- Feet, joint and LIO streams lag the IMU header clock by about 0.06 s; they were shifted onto the IMU clock.
- The LIO frame matches the IMU/body frame to within about 0.5°. The body origin offset was estimated by requiring stance feet to stay still.
- Noise floor: a planted foot "moves" about 1.2 cm per 0.1 s in the median case, which is mostly reference noise. Treat the labels as noisy.
