# Final Project: results

## Setup
- **Data:** Co-RaL, 6 sequences. Train: building, under, garage, narrow (last 20% of each held out for validation). Test: **stair, trail**.
- **Reference:** Co-RaL `gt_odom_lio` (10 Hz), converted to Spot's body frame (`results/calib.json`). LIO stamps lag the IMU by 0.06 s; the feet/joint streams are only +5 ms off the IMU.
- **Labels (v2):** per stance foot, per IMU step: how far the foot moves in the world over a centred 0.1 s window, using a **fused LIO + IMU body reference** (`code/reference.py`). `score = exp(-d² / 2·(2 cm)²)`.
- **EKF:** the Project 2b foothold EKF (filter code unchanged), now run with **continuous foot updates** (every planted foot at 25 Hz, settings tuned on training data only). Each foot gets its own foothold process noise: `σ_f,i = σ_min^(s_i) · σ_max^(1−s_i)`. σ_max is chosen per method on the training sequences only.
- **Robustness:** every number is mean ± std over 5 foot-timing offsets (−40…+40 ms).

## 1. What changed since the first version, and why

| | v1 (first results) | v2 (now) | Why |
|---|---|---|---|
| Foot updates in the EKF | only at touchdown events | continuous, 25 Hz | With touchdown-only updates a planted foot informs the filter once or twice per stance, so per-foot trust had almost nothing to act on. |
| Label reference | LIO poses only | LIO (slow part) + IMU (fast part) | LIO jitters at Spot's gait frequencies; that jitter looked like foot motion. |
| Feet stream timing | +0.06 s (assumed same clock as LIO) | **+0.005 s** (measured) | v1's assumption was wrong; it inflated labels and is why applying the shift in the EKF made it worse. |
| Label window | 10 Hz LIO intervals | sliding 0.1 s window at 100 Hz | 10× more labels, no gaps around touchdown. |
| Synthetic-slip experiment | not finished | finished | |

## 2. Label quality (`results/label_quality.json`)
"Clean" = real data. "Synthetic" = the known slips of experiment B injected into real data, so the right answer is known.

| | v1 labels | v2 labels |
|---|---|---|
| Median apparent motion of a planted foot per 0.1 s (clean) | 1.30 cm | **0.89 cm** |
| 99th percentile (clean) | 4.8 cm | **2.9 cm** |
| Clean steps wrongly scored < 0.5 ("false alarms") | 14% | **2.8%** |
| Injected slips > 2 cm caught (score < 0.5) | 95% | 95% |
| AUC, slip > 2 cm vs not | 0.967 | **0.993** |
| Correlation with true injected motion | 0.77 | **0.89** |

The fused reference removes most of the fake motion without hiding real slips (`figures/label_histogram_real.png`).
What is left (~1 cm, mostly sideways) is shared by both planted feet, so it is still reference error rather than slip.

## 3. Label prediction (test sequences)
Correlation r with the label (MAE in brackets).

| | A: real labels, stair | A: real, trail | B: synthetic, stair | B: synthetic, trail |
|---|---|---|---|---|
| Heuristic | 0.24 (0.089) | 0.22 (0.094) | 0.25 (0.051) | 0.26 (0.053) |
| MLP | 0.36 (0.090) | 0.26 (0.097) | 0.90 (0.016) | 0.89 (0.019) |
| **LSTM-MLP** | **0.47 (0.082)** | **0.36 (0.091)** | **0.95 (0.011)** | **0.96 (0.013)** |

The LSTM is the best predictor in both experiments, as hypothesized (`figures/label_prediction_test.png`).
On real data the error dropped by ~40% vs v1 (MAE 0.15 → 0.09), but r stayed low: the cleaner labels have almost no real slips left to predict.

## 4. Experiment A: real data (test mean, stair + trail)
| Method | ATE [m] | RPE /10 m [m] | Drift [%] |
|---|---|---|---|
| Original EKF (Project 2b replay, touchdown updates) | 1.53 ± 0.39 | 0.753 ± 0.231 | 2.44 ± 0.79 |
| **EKF, continuous updates** | 1.48 ± 0.55 | **0.305 ± 0.024** | 2.23 ± 1.02 |
| + Heuristic | 1.46 ± 0.54 | 0.307 ± 0.024 | 2.20 ± 1.00 |
| + MLP | 1.46 ± 0.54 | 0.305 ± 0.024 | 2.20 ± 1.00 |
| + LSTM-MLP | 1.46 ± 0.55 | 0.304 ± 0.025 | 2.19 ± 1.01 |
| + Oracle (true labels) | 1.48 ± 0.57 | 0.306 ± 0.024 | 2.22 ± 1.03 |

Per sequence: `results/summary_real.md`. Plot: `figures/ekf_metrics_real.png`.

- **Continuous updates cut the error per 10 m by 2.5×** (0.75 → 0.31 m) and halve the stair drift. This is the biggest single improvement in the project.
- **Reliability scores still change nothing on real data, not even the oracle.** The training-set sweep picks the smallest σ_max (0.03), i.e. "barely use the scores"; larger values only hurt (oracle, train RPE: 0.305 baseline → 0.315 / 0.358 / 0.418 / 0.494 for σ_max 0.03 / 0.1 / 0.3 / 1; `results/sigmax_sweep_oracle_train.json`). Spot simply does not slip much on these recordings, so there is nothing for per-foot trust to fix.
- **Trail's ATE got worse with continuous updates (1.37 → 1.93 m) while its RPE improved.** That is heading (yaw) drift, which leg contacts cannot observe; the max yaw error grew from 4.9° to 6.2°. Stair improved on every metric (ATE 1.68 → 1.03 m).

## 5. Experiment B: synthetic slips (test mean, stair + trail)
In 20% of stances the measured foot slides 3–10 cm backward. **Slip damage** = RMS distance between the run on slipped data and the same filter's run on clean data (isolates slip error from heading drift).

| Method | Slip damage RMS [m] | ATE [m] | RPE /10 m [m] | Drift [%] |
|---|---|---|---|---|
| Original EKF (touchdown updates) | 2.58 ± 0.32 | 2.09 ± 0.35 | 0.887 ± 0.234 | 2.45 ± 0.82 |
| EKF, continuous updates | 3.21 ± 0.37 | 2.41 ± 0.39 | 0.610 ± 0.066 | 2.86 ± 0.91 |
| + Heuristic | 2.38 ± 0.41 | 2.11 ± 0.40 | 0.537 ± 0.052 | 2.57 ± 0.93 |
| + MLP | **0.62 ± 0.14** | 1.43 ± 0.51 | **0.343 ± 0.035** | 2.12 ± 0.99 |
| + LSTM-MLP | 0.63 ± 0.13 | **1.41 ± 0.51** | **0.343 ± 0.034** | **2.08 ± 0.99** |
| + Oracle (true labels) | 0.61 ± 0.19 | 1.40 ± 0.54 | 0.345 ± 0.034 | 2.07 ± 1.03 |

Per sequence: `results/summary_synth.md`. Plot: `figures/ekf_metrics_synth.png`.

- **When slips exist, learned reliability works.** The MLP and LSTM cut slip damage by **80%** (3.2 → 0.6 m) and recover ~88% of the RPE lost to slips (clean-data RPE is 0.305).
- **Learned ≈ oracle.** Both networks reach the same EKF accuracy as perfect labels.
- **LSTM vs MLP:** the LSTM predicts labels better (r 0.95 vs 0.90), but both are already good enough that the EKF result is the same.
- **The heuristic barely helps** (r ≈ 0.25; damage 3.2 → 2.4 m).
- Continuous updates make the plain filter *more* sensitive to slips (damage 2.6 → 3.2 m), because a slipping foot now drags the body all through the stance. That is exactly the situation per-foot trust is for: with it, the continuous filter beats the original by 4× on slip damage.

## 6. Bottom line
1. Temporal context (LSTM) gives the best reliability predictions, on real and synthetic labels.
2. On real Co-RaL data, foot reliability cannot improve the EKF, even with perfect labels, because there is very little slip. The remaining error is heading drift.
3. When slips are present (synthetic), learned reliability removes ~80% of the slip damage and matches the oracle.
4. A large, method-independent gain came from running the EKF with continuous foot updates (RPE 0.75 → 0.31 m per 10 m).

## Caveats
- LIO is both the label source (fused with IMU) and the evaluation reference.
- Synthetic slips only corrupt the reported foot position; a real slip would also change the body dynamics and joint signals.
- One training seed. σ_max grid {0.03, 0.1, 0.3, 1}. The heuristic hit the top of its σ_max grid in experiment B.
- Superseded v1 logs: `results/sigmax_sweep_legacy_touchdown.log` (touchdown filter, v1 labels, also tried R-scaling and dropping feet; all worse).

## Folder map
- `code/`: everything. Pipeline: `run_all.sh` (extract → calib → tune_ekf → features → label_quality → synth → models → experiment → eval_ekf → analyze → fig_labels)
- `extracted/`: raw arrays from the bags
- `gt/`: Co-RaL LIO
- `dataset/`: team-format files (see its README)
- `features/`, `results/`, `figures/`
