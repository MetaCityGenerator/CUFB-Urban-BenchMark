#!/usr/bin/env bash
# Re-aggregate without rerunning analyze: bounds + bootstrap, type/size strata, wide and long tables, archetypes, test reference and controlled-test rescoring, generator rescoring, results table, manifest.
# Used after individual counties are recomputed (e.g. after the land-use layer of a few counties was replaced). Era strata, reference spectrum and building-independent parts are untouched.
# Usage: bash run_reaggregate.sh   (from the repository root; ~20-30 min, mostly the 200 bootstrap resamples)
set -u
export PYTHONIOENCODING=utf-8
B=code/urbanBenchmark; O=dataset/output_county; REF=dataset/ref_holdout
echo "=== [1/7] national bounds + bootstrap 200  $(date +%H:%M) ==="
python $B/benchmark.py aggregate --root $O --min-cities 20 --min-n 1000 --bootstrap 200
echo "=== [2/7] strata type / size  $(date +%H:%M) ==="
python $B/benchmark.py aggregate --root $O --min-cities 20 --min-n 1000 --strata type --input-root dataset/input_county
python $B/benchmark.py aggregate --root $O --min-cities 20 --min-n 1000 --strata size
echo "=== [3/7] tables  $(date +%H:%M) ==="
python $B/export.py --root $O --input-root dataset/input_county
python $B/export.py --root $O --relations --input-root dataset/input_county
python $B/archetypes.py --root $O
echo "=== [4/7] holdout reference + gate-1 rescoring  $(date +%H:%M) ==="
python $B/benchmark.py aggregate --root $O --min-cities 20 --min-n 1000 --holdout $REF/holdout.txt --out-dir $REF
python $B/score.py --real dataset/holdout_out --perturbed dataset/perturbed_out --ref $REF --protocol v2 --out $REF/scores_v2.csv | tail -12
python $B/score.py --real dataset/holdout_out --perturbed dataset/perturbed_out --ref $REF --protocol v1 --out $REF/scores_v1.csv | tail -3
echo "=== [5/7] generator cases rescored against national reference  $(date +%H:%M) ==="
for d in dataset/generated/*/; do
  [ -f "$d/result.json" ] || continue
  python $B/score.py --city "${d%/}" --ref $O --protocol v2 > /dev/null || echo "  [ERR] $d"
done
echo "=== [6/7] benchmark_results + manifest  $(date +%H:%M) ==="
python $B/export.py --root $O --results
python $B/export.py --root $O --manifest --version 1.0
echo "=== [7/7] figures  $(date +%H:%M) ==="
python $B/fig_paper.py --only fig2,fig3,fig4,fig5,fig7 | grep -v Warn
python $B/fig_paper.py --gis | tail -1
echo "=== REAGGREGATE DONE $(date +%H:%M) ==="
