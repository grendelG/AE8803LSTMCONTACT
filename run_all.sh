#!/usr/bin/env bash
# Full pipeline from the extracted arrays (extracted/*.npz) to tables + figures.
# Run from the repo root:  bash run_all.sh
# Needs numpy, scipy, gtsam, torch, matplotlib. Times are for a 20-core CPU.
set -e
cd code
python calib.py              # LIO frame/timing + feet timing calibration -> results/calib.json        (~1 min)
python tune_ekf.py           # continuous-update EKF settings, TRAIN only -> results/ekf_tuning.json   (~20 min)
python features.py           # features + labels (fused LIO+IMU ref)    -> features/*.npz             (~1 min)
python label_quality.py      # legacy vs current labels                 -> results/label_quality.json (~1 min)
python export_dataset.py     # team-format files                        -> dataset/*.npz
python synth.py              # synthetic-slip features + exact labels   -> features_synth/*.npz
python models.py             # heuristic / MLP / LSTM-MLP on real labels  -> results/scores/, *.pt     (~10 min)
EXP=synth python models.py   # same, on synthetic labels                -> results/scores_synth/      (~10 min)
python experiment.py         # oracle sweep of sigma_max on TRAIN         -> results/sigmax_sweep_oracle_train.json
python eval_ekf.py real  baseline_td baseline heuristic mlp lstm oracle            # -> results/ekf_real/
python eval_ekf.py synth baseline_td baseline heuristic mlp lstm oracle            # -> results/ekf_synth/
python analyze.py real       # tables + bar chart -> results/summary_real.*, figures/
python analyze.py synth      #                    -> results/summary_synth.*, figures/
python fig_labels.py         # label figures      -> figures/
