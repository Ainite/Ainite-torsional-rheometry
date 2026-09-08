from pathlib import Path
import json, sys
R=Path(__file__).resolve().parent/'results'; d=json.load(open(R/'R181_CROSS_REPLICATION_GATE.json'))
assert d['verdict']=='CANCELLATION_REPLICATED_LOCATION_NOT_REPLICATED'
assert d['subgates']['cancellation_pass_count']==2
assert d['subgates']['location_pass_count']==0
for c in d['cases']:
    assert c['valid']
    assert c['cancellation']['orth_angle_deg']>150
    assert c['cancellation']['reduction_fraction']>=.5
    assert max([c['full']['step_angle_deg']]+[x['step_angle_deg'] for x in c['interface_step_checks']]+[x['step_angle_deg'] for x in c['window_rows']])<.01
print('R181 VERIFY PASS')
