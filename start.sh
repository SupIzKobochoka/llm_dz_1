#!/usr/bin/env bash
set -e

python experiment_01_baseline.py
python experiment_02_large_lr.py
python experiment_03_small_lr.py
python experiment_04_small_batch.py
python experiment_05_large_batch.py
