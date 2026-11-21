## Experiment A: real LIO labels (test = stair, trail)

mean ± std over 5 foot-timing offsets (−40…+40 ms); σ_max chosen on training sequences by RPE

| Method | σ_max | Seq | ATE [m] | RPE /10 m [m] | Drift [%] |
|---|---|---|---|---|---|
| Original EKF (2b replay) | — | stair | 1.68 ± 0.41 | 0.976 ± 0.078 | 2.86 ± 0.85 |
| Original EKF (2b replay) | — | trail | 1.37 ± 0.29 | 0.530 ± 0.032 | 2.02 ± 0.44 |
| Original EKF (2b replay) | — | mean_test | 1.53 ± 0.39 | 0.753 ± 0.231 | 2.44 ± 0.79 |
| EKF, continuous updates | — | stair | 1.03 ± 0.27 | 0.306 ± 0.027 | 1.40 ± 0.47 |
| EKF, continuous updates | — | trail | 1.93 ± 0.35 | 0.303 ± 0.020 | 3.05 ± 0.69 |
| EKF, continuous updates | — | mean_test | 1.48 ± 0.55 | 0.305 ± 0.024 | 2.23 ± 1.02 |
| Heuristic | 0.03 | stair | 1.02 ± 0.29 | 0.308 ± 0.025 | 1.39 ± 0.51 |
| Heuristic | 0.03 | trail | 1.91 ± 0.32 | 0.306 ± 0.023 | 3.00 ± 0.65 |
| Heuristic | 0.03 | mean_test | 1.46 ± 0.54 | 0.307 ± 0.024 | 2.20 ± 1.00 |
| MLP | 0.03 | stair | 1.01 ± 0.29 | 0.305 ± 0.023 | 1.39 ± 0.52 |
| MLP | 0.03 | trail | 1.92 ± 0.31 | 0.305 ± 0.024 | 3.01 ± 0.64 |
| MLP | 0.03 | mean_test | 1.46 ± 0.54 | 0.305 ± 0.024 | 2.20 ± 1.00 |
| LSTM-MLP | 0.03 | stair | 1.01 ± 0.31 | 0.305 ± 0.026 | 1.37 ± 0.53 |
| LSTM-MLP | 0.03 | trail | 1.92 ± 0.31 | 0.304 ± 0.024 | 3.02 ± 0.63 |
| LSTM-MLP | 0.03 | mean_test | 1.46 ± 0.55 | 0.304 ± 0.025 | 2.19 ± 1.01 |
| Oracle labels | 0.03 | stair | 1.02 ± 0.29 | 0.309 ± 0.027 | 1.39 ± 0.51 |
| Oracle labels | 0.03 | trail | 1.95 ± 0.34 | 0.303 ± 0.021 | 3.06 ± 0.68 |
| Oracle labels | 0.03 | mean_test | 1.48 ± 0.57 | 0.306 ± 0.024 | 2.22 ± 1.03 |