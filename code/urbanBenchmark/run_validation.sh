#!/usr/bin/env bash
# Technical Validation in one run (from the repository root; needs runall and dataset/ref_holdout/holdout.txt from run_gate1.sh)
# Usage: bash run_validation.sh [N_CROSS=50] [N_SENS=20] [N_BOOT=20] [REPS=20]
# Output: dataset/validation/{crosscheck_counties.csv, crosscheck_summary.json, sensitivity/, sensitivity_summary.csv,
#         bootstrap/, bootstrap_summary.csv}; *_ci_lo/*_ci_hi columns added to dataset/output_county/national_bounds.csv
set -u
NC=${1:-50}; NS=${2:-20}; NB=${3:-20}; REPS=${4:-20}
export PYTHONIOENCODING=utf-8
B=code/urbanBenchmark; V=dataset/validation; L=dataset/ref_holdout/holdout.txt
mkdir -p $V
echo "=== [1/4] crosscheck momepy/osmnx on $NC counties  $(date +%H:%M) ==="
python $B/validate.py crosscheck --root dataset/input_county --counties $L --n $NC --out $V | grep -vE "^\s+\[[0-9]+/"
echo "=== [2/4] national bounds with province block bootstrap x200  $(date +%H:%M) ==="
python $B/benchmark.py aggregate --root dataset/output_county --min-cities 20 --min-n 1000 --bootstrap 200
echo "=== [3/4] parameter sensitivity on $NS counties  $(date +%H:%M) ==="
python $B/validate.py sensitivity --root dataset/input_county --counties $L --n $NS --out $V | grep -vE "^\s+\[[0-9]+/"
echo "=== [4/4] building subsample bootstrap on $NB counties x $REPS  $(date +%H:%M) ==="
python $B/validate.py bootstrap --root dataset/input_county --counties $L --n $NB --reps $REPS --frac 0.8 --out $V | grep -vE "^\s+\[[0-9]+/"
echo "=== VALIDATION DONE $(date +%H:%M) ==="
