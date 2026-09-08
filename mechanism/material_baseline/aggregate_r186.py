#!/usr/bin/env python3
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent; OUT=HERE/'results'
rule=json.loads((OUT/'R186_PREREGISTERED_RULE.json').read_text())
cases=[json.loads((OUT/f'case_f_{f:.2f}.json').read_text()) for f in rule['new_f']]
if not all(c['valid'] for c in cases): verdict='INVALID_NUMERICAL_GATE'
elif all(c['pass'] for c in cases): verdict='MATERIAL_BASELINE_NESTED_CANCELLATION_REPLICATED'
else: verdict='MATERIAL_BASELINE_REPLICATION_FAILED'
out={'schema':'U1-R186-material-baseline-v1','preregistered_rule':rule,'cases':cases,'verdict':verdict}
(OUT/'R186_MATERIAL_BASELINE_GATE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'verdict':verdict,'passes':[c['pass'] for c in cases]},indent=2))
