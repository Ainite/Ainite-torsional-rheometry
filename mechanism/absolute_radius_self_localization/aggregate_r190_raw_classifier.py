#!/usr/bin/env python3
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent; OUT=HERE/'results'; rule=json.loads((OUT/'R190_PREREGISTERED_RULE.json').read_text())
cases=[json.loads((OUT/f'R190_CASE_XMAX_{x:.2f}.json').read_text()) for x in rule['new_observer_xmax']]
if all(c['pass'] for c in cases): verdict='PROSPECTIVE_OBSERVER_SELF_LOCALIZATION_SUPPORTED'
elif any(not c['valid'] for c in cases): verdict='INVALID_NUMERICAL_GATE'
else: verdict='PROSPECTIVE_OBSERVER_SELF_LOCALIZATION_NOT_SUPPORTED'
out={'schema':'U1-R190-prospective-self-localization-v1','preregistered_rule':rule,'cases':cases,'verdict':verdict}
(OUT/'R190_PROSPECTIVE_SELF_LOCALIZATION_GATE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'verdict':verdict,'summary':[{'xmax':c['xmax'],'winner':c['winner']['midpoint'] if c['winner'] else None,'distance':c['winner']['distance_to_xmax'] if c['winner'] else None,'reduction':c['winner']['reduction_fraction'] if c['winner'] else None,'pass':c['pass']} for c in cases]},indent=2))
