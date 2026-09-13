#!/usr/bin/env bash
# Reproduce every number and figure in the README.
# Parts 1-3 are minutes. Parts 4-5 are the sweeps: ~45 min and ~95 min on an M-series
# GPU (MPS), well under that on a CUDA card. Device is picked automatically.
set -e
mkdir -p results figures data
[ -f data/tinyshakespeare.txt ] || curl -sL -o data/tinyshakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

PY=${PY:-python3}
$PY -u exp1_adam_by_hand.py   2>&1 | tee results/exp1_adam_by_hand.txt    # parts 1 and 2
$PY -u exp2_update_ratios.py  2>&1 | tee results/exp2_update_ratios.txt   # part 3
$PY -u exp3_schedules.py      2>&1 | tee results/exp3_schedules.txt       # part 4
$PY -u exp4_lr_width_sweep.py 2>&1 | tee results/exp4_lr_width.txt        # part 5
