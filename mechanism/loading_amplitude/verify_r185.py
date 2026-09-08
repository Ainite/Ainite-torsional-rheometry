from pathlib import Path
import json
R=Path(__file__).resolve().parent/'results'; d=json.load(open(R/'R185_LOADING_AMPLITUDE_GATE.json'))
assert d['verdict']=='LOADING_AMPLITUDE_NESTED_CANCELLATION_REPLICATED'
assert len(d['cases'])==2
for c in d['cases']:
    assert c['valid'] and c['pass'] and c['level1']['pass'] and c['level2']['pass']
    assert c['level1']['direct_state_angle_deg']>160
    assert .75<=c['level1']['direct_over_state']<=1.33
    assert c['level1']['total_over_max_component']<=.35
    assert c['level2']['total_band_angle_to_inner_deg']>150
    assert c['level2']['reduction_fraction']>=.5
print('R185 VERIFY PASS')
