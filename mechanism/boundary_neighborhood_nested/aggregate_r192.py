#!/usr/bin/env python3
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';rule=json.loads((OUT/'R192_PREREGISTERED_RULE.json').read_text());cases=[json.loads((OUT/f'R192_CASE_XMAX_{x:.2f}.json').read_text()) for x in rule['new_observer_xmax']]
if all(c['pass'] for c in cases): verdict='PROSPECTIVE_BOUNDARY_NEIGHBORHOOD_NESTED_CANCELLATION_SUPPORTED'
elif any(not c['numerical_valid'] for c in cases): verdict='INVALID_NUMERICAL_GATE'
else: verdict='PROSPECTIVE_BOUNDARY_NEIGHBORHOOD_NESTED_CANCELLATION_NOT_SUPPORTED'
out={'schema':'U1-R192-boundary-neighborhood-v1','preregistered_rule':rule,'cases':cases,'verdict':verdict};(OUT/'R192_BOUNDARY_NEIGHBORHOOD_GATE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'verdict':verdict,'cases':[{'xmax':c['xmax'],'mids':c['preselected_midpoints'],'passes':[b['band_pass'] for b in c['bands']],'case_pass':c['pass']} for c in cases]},indent=2))
