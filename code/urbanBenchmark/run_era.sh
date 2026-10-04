#!/usr/bin/env bash
# Construction-era bounds: run a buildings-only analyze for every county per construction era (CMAB Age field), then aggregate --strata era.
# Usage: bash run_era.sh [ERAS="pre1990 1991_2000 2001_2010 post2010"]   (from the repository root; runall must be finished)
# Output: dataset/output_county/{county}/result_{era}.json + spectrum_{era}.json, dataset/output_county/national_bounds_by_era.csv, manifest.json
# Resumable: counties with an existing result_{era}.json are skipped; rerun the same command after an interruption. About 2-4 s per county (building blocks only), ~8 h for 4 eras.
set -u
ERAS=${1:-"pre1990 1991_2000 2001_2010 post2010"}
export PYTHONIOENCODING=utf-8
B=code/urbanBenchmark; OUT=dataset/output_county
for e in $ERAS; do
  echo "=== era $e  $(date '+%m-%d %H:%M') ==="
  python $B/benchmark.py runall --root dataset/input_county --out $OUT --era $e --no-fig | grep -E --line-buffered "^\[runall\]|\[ERR\]|^ +\[[0-9]*(00|50)/"
done
echo "=== aggregate --strata era  $(date '+%m-%d %H:%M') ==="
python $B/benchmark.py aggregate --root $OUT --min-cities 20 --min-n 1000 --strata era
python $B/export.py --root $OUT --manifest --version 1.0-dev
echo "=== ERA DONE $(date '+%m-%d %H:%M') ==="
