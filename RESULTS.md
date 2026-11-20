# Final Project: results so far

## Setup
- **Data:** Co-RaL, 6 sequences. Train: building, under, garage, narrow (last 20% of each held out for validation). Test: **stair, trail**.
- **Reference:** Co-RaL `gt_odom_lio` (10 Hz), converted to Spot's body frame (`results/calib.json`: rotation < 1°, LIO/feet streams ~60 ms behind IMU stamps).
- **Labels:** per stance foot, per 0.1 s: foot displacement in the world (LIO pose × body→foot). `score = exp(-d² / 2·(2 cm)²)`.
- **EKF:** the Project 2b foothold EKF, unchanged except that each foot gets its own foothold process noise: `σ_f,i = 1e-4^(s_i) · σ_max^(1−s_i)`. σ_max is chosen per method on the training sequences only. Height prior off.
- **Robustness:** every number is mean ± std over 5 foot-timing offsets (−40…+40 ms). The EKF is very sensitive to timing; a 20 ms change moved ATE on `under` from 2.55 m to 0.78 m.

## 1. Label prediction (test sequences, correlation r with true label)
| | stair | trail |
|---|---|---|
| Heuristic | 0.33 | 0.27 |
| MLP | 0.43 | 0.37 |
| **LSTM-MLP** | **0.46** | **0.41** |

The LSTM matches the labels best, as hypothesized. (`figures/label_prediction_test.png`)

## 2. State estimation (test mean)
| Method | ATE [m] | RPE /10 m [m] | Drift [%] |
|---|---|---|---|
| Original EKF | 1.53 ± 0.39 | 0.753 ± 0.231 | 2.44 ± 0.79 |
| Heuristic | 1.53 ± 0.39 | 0.753 ± 0.232 | 2.45 ± 0.80 |
| MLP | 1.53 ± 0.39 | 0.753 ± 0.231 | 2.44 ± 0.79 |
| LSTM-MLP | 1.53 ± 0.39 | 0.753 ± 0.231 | 2.44 ± 0.79 |
| Oracle (true labels) | 1.52 ± 0.39 | 0.753 ± 0.231 | 2.43 ± 0.79 |

Per-sequence table: `results/summary_real.md`. Plot: `figures/ekf_metrics_real.png`.

**No method changes the EKF, not even perfect labels.** Why:
1. **The labels are mostly reference noise.** A planted foot "moves" 1.3 cm per 0.1 s in the median case, in a single hump with a weak tail (`figures/label_histogram_real.png`). With a trot, only 2 feet are down at a time, so you can't tell one slipping foot from LIO jitter.
2. **The baseline error is heading drift, not slip.** Distance traveled is accurate to about 1%, but yaw drifts (24° over building). Foot reliability can't fix yaw.
3. On training data, scaling up the noise for low-score feet (either foothold or measurement noise), or dropping those feet entirely, all made the EKF slightly *worse* (`results/sigmax_sweep.log`).

## Not finished: synthetic slips (the proposal's fallback)
Code is in `code/synth.py`. Backward push-off slips (3–10 cm, 20% of stances) pull the baseline trajectory about 2.9 m (stair) and 2.3 m (trail) RMS away from the clean run; perfect labels cut that to about 1.2 m and 0.7 m (one-off check, offset 0 only). Models were not retrained on the fixed labels and no comparison was run. Re-run: `python synth.py`, `EXP=synth python models.py`, `python eval_ekf.py synth baseline heuristic mlp lstm oracle`, `python analyze.py synth`. This needs `results/ekf_real/traj/`, which is produced by `python eval_ekf.py real baseline`.

## Folder map
- `code/`: everything. Pipeline order: extract → calib → features → models → eval_ekf → analyze
- `extracted/`: raw arrays from the bags
- `gt/`: Co-RaL LIO
- `dataset/`: team-format files (see its README)
- `features/`, `results/`, `figures/`
