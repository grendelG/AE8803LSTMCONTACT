## Experiment A: real LIO labels (test = stair, trail)

mean ± std over 5 foot-timing offsets (−40…+40 ms); σ_max chosen on training sequences by RPE

| Method | σ_max | Seq | ATE [m] | RPE /10 m [m] | Drift [%] |
|---|---|---|---|---|---|
| Original EKF | — | stair | 1.68 ± 0.41 | 0.976 ± 0.078 | 2.86 ± 0.85 |
| Original EKF | — | trail | 1.37 ± 0.29 | 0.530 ± 0.032 | 2.02 ± 0.44 |
| Original EKF | — | mean_test | 1.53 ± 0.39 | 0.753 ± 0.231 | 2.44 ± 0.79 |
| Heuristic | 0.03 | stair | 1.69 ± 0.41 | 0.977 ± 0.077 | 2.86 ± 0.85 |
| Heuristic | 0.03 | trail | 1.37 ± 0.29 | 0.529 ± 0.033 | 2.03 ± 0.44 |
| Heuristic | 0.03 | mean_test | 1.53 ± 0.39 | 0.753 ± 0.232 | 2.45 ± 0.80 |
| MLP | 0.01 | stair | 1.68 ± 0.41 | 0.976 ± 0.078 | 2.86 ± 0.85 |
| MLP | 0.01 | trail | 1.37 ± 0.29 | 0.530 ± 0.032 | 2.02 ± 0.44 |
| MLP | 0.01 | mean_test | 1.53 ± 0.39 | 0.753 ± 0.231 | 2.44 ± 0.79 |
| LSTM-MLP | 0.01 | stair | 1.68 ± 0.41 | 0.976 ± 0.078 | 2.86 ± 0.85 |
| LSTM-MLP | 0.01 | trail | 1.37 ± 0.29 | 0.530 ± 0.032 | 2.02 ± 0.44 |
| LSTM-MLP | 0.01 | mean_test | 1.53 ± 0.39 | 0.753 ± 0.231 | 2.44 ± 0.79 |
| Oracle labels | 0.01 | stair | 1.68 ± 0.41 | 0.976 ± 0.078 | 2.85 ± 0.85 |
| Oracle labels | 0.01 | trail | 1.37 ± 0.29 | 0.530 ± 0.033 | 2.01 ± 0.44 |
| Oracle labels | 0.01 | mean_test | 1.52 ± 0.39 | 0.753 ± 0.231 | 2.43 ± 0.79 |