from pathlib import Path
import json
R=Path(__file__).resolve().parent/'results'
rule=json.load(open(R/'R181_PREREGISTERED_RULE.json'))
A=json.load(open(R/'case_aspect5_independent_radial.json'))
B=json.load(open(R/'case_aspect2p5_independent_geometry.json'))
can=[A['cancellation']['pass'],B['cancellation']['pass']]
loc=[A['collapse_location']['pass_preinterface_location'],B['collapse_location']['pass_preinterface_location']]
valid=A['valid'] and B['valid']
if not valid: verdict='INVALID_NUMERICAL_GATE'
elif all(can) and all(loc): verdict='COMBINED_CANCELLATION_AND_LOCATION_REPLICATED'
elif all(can) and not any(loc): verdict='CANCELLATION_REPLICATED_LOCATION_NOT_REPLICATED'
elif all(can): verdict='CANCELLATION_REPLICATED_LOCATION_MIXED'
else: verdict='CANCELLATION_NOT_REPLICATED'
out={'schema':'U1-R181-cross-replication-v2','preregistered_rule':rule,'feasibility_amendment':'R181_FEASIBILITY_AMENDMENT.md','original_34x40_status':'unresolved for computational feasibility; two runs produced no scientific output before timeout','cases':[A,B],'subgates':{'cancellation_passes':can,'cancellation_pass_count':sum(can),'location_passes':loc,'location_pass_count':sum(loc)},'verdict':verdict,'claim_boundary':'The annular-cancellation vector mechanism replicates across an independent bulk radial partition and a second thickness on the amended 34x20 grid. The radial location of the largest local collapse does not replicate, and full axial mesh independence is not established.'}
(R/'R181_CROSS_REPLICATION_GATE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'verdict':verdict,'subgates':out['subgates']},indent=2))
