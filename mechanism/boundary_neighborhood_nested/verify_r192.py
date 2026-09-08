#!/usr/bin/env python3
import json
from pathlib import Path
d=json.loads((Path(__file__).resolve().parent/'results'/'R192_BOUNDARY_NEIGHBORHOOD_GATE.json').read_text());assert len(d['cases'])==3
for c in d['cases']:
 assert c['numerical_valid']; assert c['max_step_angle_deg']<.01
 for b in c['bands']: assert b['closure_over_max_component']<1e-5
print('R192 VERIFY PASS:',d['verdict'])
