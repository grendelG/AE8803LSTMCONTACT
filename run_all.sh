#!/usr/bin/env bash
# Full pipeline from the extracted arrays (extracted/*.npz) to tables + figures.
# Run from the repo root:  bash run_all.sh
set -e
cd code
python calib.py              # frame + timing calibration   -> results/calib.json            (~10 s)
python features.py           # features + LIO labels         -> features/*.npz                 (~10 s)
python export_dataset.py     # team-format files             -> dataset/*.npz                  (~10 s)
python models.py             # heuristic / MLP / LSTM-MLP    -> results/scores/, *.pt          (~5-10 min CPU)
python experiment.py         # oracle sweep of sigma_max on training sequences                 (~5 min)
python eval_ekf.py real baseline heuristic mlp lstm oracle   # all EKF runs -> results/ekf_real/ (~30-40 min)
python analyze.py real       # tables + bar chart            -> results/summary_real.*, figures/
python fig_labels.py         # label figures                 -> figures/
