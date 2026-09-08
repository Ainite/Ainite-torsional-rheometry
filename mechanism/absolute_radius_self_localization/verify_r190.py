#!/usr/bin/env python3
import json
from pathlib import Path
p=Path(__file__).resolve().parent/'results'/'R190_PROSPECTIVE_SELF_LOCALIZATION_GATE.json'
d=json.loads(p.read_text())
assert len(d['cases'])==3
for c in d['cases']:
    assert c['max_step_angle_deg']<.01
    assert c['baseline_grad_inf_pre']<1e-7 and c['baseline_grad_inf_twist']<1e-7
    assert c['winner_mechanics'] is not None
    assert c['winner_mechanics']['closure_over_max_component']<1e-5
    assert c['localization_pass']
assert d['verdict']=='PROSPECTIVE_OBSERVER_SELF_LOCALIZATION_NOT_SUPPORTED'
assert sum(bool(c['mechanics_pass']) for c in d['cases'])==1
print('R190 VERIFY PASS:',d['verdict'])
for c in d['cases']:
    print(c['xmax'],c['winner']['midpoint'],c['winner']['reduction_fraction'],c['winner_mechanics']['direct_state_angle_deg'],c['winner_mechanics']['direct_over_state'],c['mechanics_pass'])
