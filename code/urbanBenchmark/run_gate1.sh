#!/usr/bin/env bash
# Controlled test: hold out N random counties, recompute the bounds from the others, apply 7 perturbations to the held-out counties and compare real vs perturbed scores (AUC).
# Usage: bash run_gate1.sh [N=50] [SEED=0]      (from the repository root; benchmark.py runall must be finished)
# Output: dataset/ref_holdout/{holdout.txt, national_bounds.csv, national_reference_spectrum.*, spectrum_deviation.csv, scores_v2.csv, scores_v1.csv}
#         dataset/perturbed/ (perturbed inputs, can be deleted), dataset/perturbed_out/ (perturbed results), dataset/holdout_out/ (copies of held-out results)
set -u
N=${1:-50}; SEED=${2:-0}
export PYTHONIOENCODING=utf-8
B=code/urbanBenchmark; REF=dataset/ref_holdout
mkdir -p $REF
echo "=== [1/5] perturb $N counties x 7 kinds  $(date +%H:%M) ==="
python $B/perturb.py --input dataset/input_county --n $N --min-n 1000 --seed $SEED --list $REF/holdout.txt --out dataset/perturbed --kinds all | tail -2
tr -d '\r' < $REF/holdout.txt > $REF/holdout.lf && mv $REF/holdout.lf $REF/holdout.txt
echo "=== [2/5] holdout bounds + reference spectrum  $(date +%H:%M) ==="
python $B/benchmark.py aggregate --root dataset/output_county --min-cities 20 --min-n 1000 --holdout $REF/holdout.txt --out-dir $REF
python $B/spectrum.py reference --root dataset/output_county --min-n 1000 --holdout $REF/holdout.txt --out-dir $REF | grep -v -E "^(largest|smallest) deviation"
echo "=== [3/5] analyze perturbed  $(date +%H:%M) ==="
python $B/benchmark.py runall --root dataset/perturbed --out dataset/perturbed_out --no-fig | grep -E "^\[runall\]|\[ERR\]"
echo "=== [4/5] collect real holdout results  $(date +%H:%M) ==="
rm -rf dataset/holdout_out; mkdir -p dataset/holdout_out
while IFS= read -r c; do
  [ -n "$c" ] || continue
  mkdir -p "dataset/holdout_out/$c" && cp "dataset/output_county/$c/result.json" "dataset/output_county/$c/spectrum.json" "dataset/holdout_out/$c/"
done < $REF/holdout.txt
echo "collected $(ls dataset/holdout_out | wc -l) real counties"
echo "=== [5/5] score  $(date +%H:%M) ==="
python $B/score.py --real dataset/holdout_out --perturbed dataset/perturbed_out --ref $REF --protocol v2 --out $REF/scores_v2.csv
python $B/score.py --real dataset/holdout_out --perturbed dataset/perturbed_out --ref $REF --protocol v1 --out $REF/scores_v1.csv
echo "=== GATE1 DONE $(date +%H:%M) ==="
