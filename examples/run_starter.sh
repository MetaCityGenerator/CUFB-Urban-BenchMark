#!/usr/bin/env bash
# Minimal runnable example: analyse and score the starter county released with the data record.
# 1) Download 02_reference_bounds.zip and 05_starter.zip from the Figshare record and unzip them into data/
#    (so that data/02_reference_bounds/ and data/05_starter/<county>/ exist).
# 2) From the repository root:  bash examples/run_starter.sh
# Runs in well under a minute on a laptop. The metric values in the new result.json must equal
# the released data/05_starter/<county>/result.json; the script checks this.
set -euo pipefail
B=code/urbanBenchmark
D=data
STARTER=$(find "$D/05_starter" -mindepth 1 -maxdepth 1 -type d | head -n 1)
OUT=starter_out
python "$B/benchmark.py" analyze --input "$STARTER" --out "$OUT" --no-fig
python "$B/score.py" --city "$OUT" --ref "$D/02_reference_bounds"
python - "$STARTER/result.json" "$OUT/result.json" <<'PY'
import json, sys
def flat(d, p=''):
    out = {}
    for k, v in d.items():
        out.update(flat(v, p + k + '.') if isinstance(v, dict) else {p + k: v})
    return out
a, b = (flat(json.load(open(f, encoding='utf-8'))) for f in sys.argv[1:3])
diff = [k for k in set(a) | set(b) if a.get(k) != b.get(k) and k != 'folder']
print('reproduced the released result.json' if not diff else f'differences in {len(diff)} keys: {sorted(diff)[:10]}')
sys.exit(1 if diff else 0)
PY
