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
For every IMU step and every foot that is in contact for the whole centred 0.1 s window around it, the foot is
placed in the world (body pose × body→foot) at both ends of the window and its displacement `d` is measured.
A planted foot should not move: `score = exp(-d² / (2·0.02²))` (2 cm per 0.1 s = 20 cm/s scores 0.61).
Swing feet and windows containing touchdown or liftoff get no label (`label_mask = False`).

The body pose is a **fused LIO + IMU reference** (`code/reference.py`), not raw LIO:
- velocity = slow part of LIO velocity (< 1 Hz) + fast part of IMU-integrated velocity (> 1 Hz);
- attitude = gyro integration with the gyro bias and slow drift corrected toward LIO.
LIO alone jitters at Spot's gait frequencies, which used to show up as fake foot motion.

## Timing / frames (see `results/calib.json`)
- LIO poses lag the IMU header clock by about 0.06 s (`time_offset`).
- Feet and joint streams are only about **+5 ms** off the IMU clock (`feet_dt`, measured by minimising planted-foot
  motion against the fused reference). They were shifted onto the IMU clock.
- The LIO frame matches the IMU/body frame to within about 0.5°. The body origin offset was estimated by requiring stance feet to stay still.
- Noise floor: a planted foot "moves" about 0.9 cm per 0.1 s in the median case (it was 1.3 cm with the
  older LIO-only labels). Most of what remains is a ~1 cm sideways wobble shared by all planted feet, i.e. still
  mostly reference error, not slip. Treat the labels as noisy; only about 3% of clean steps score below 0.5.
