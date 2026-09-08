from pathlib import Path
import json
R=Path(__file__).resolve().parent/'results'; d=json.load(open(R/'R184_NESTED_CANCELLATION_GATE.json'))
assert d['verdict']=='NESTED_OBSERVER_BOUNDARY_CANCELLATION_SUPPORTED'
assert d['valid'] and d['max_step_angle_deg']<.01 and d['max_decomposition_closure_rel']<1e-5
assert len(d['cases'])==3
for c in d['cases']:
    assert c['level1']['pass'] and c['level2']['pass'] and c['pass']
    assert c['level1']['direct_state_angle_deg']>160
    assert .75<=c['level1']['direct_over_state']<=1.33
    assert c['level1']['total_over_max_component']<=.35
    assert c['level2']['total_band_angle_to_inner_deg']>150
    assert c['level2']['reduction_fraction']>=.5
print('R184 VERIFY PASS')
