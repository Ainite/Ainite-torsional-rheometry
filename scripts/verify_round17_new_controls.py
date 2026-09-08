#!/usr/bin/env python3
from pathlib import Path
import json, math
ROOT=Path(__file__).resolve().parent.parent
checks={}
# source identity: corrected classifier and all three cases valid
p=ROOT/'mechanism/source_identity_control/results/R194_SOURCE_IDENTITY_CONTROL_CORRECTED.json'
d=json.loads(p.read_text())
checks['source_identity_verdict']=d.get('verdict')=='OBSERVER_GENERAL_SOURCE_IDENTITY_CONTROL'
checks['source_identity_3of3_localize']=len(d.get('cases',[]))==3 and all(c.get('localization_pass_written_rule') and c.get('numerically_valid') for c in d['cases'])
checks['source_identity_reductions']=all(c['winner']['reduction_fraction']>=0.5 for c in d['cases'])
# resolution/projection: preserve cancellation across d=0.01/0.02/0.04 and rank2/3; outer extension not boundary-limited
p=ROOT/'mechanism/resolution_projection_control/results/R195_RESOLUTION_PROJECTION_CONTROL.json'
d=json.loads(p.read_text())
checks['resolution_three_cases']=len(d.get('cases',[]))==3
checks['resolution_widths']=all(all(w['rank_results']['3']['reduction_fraction']>=0.60 for w in c['width_sensitivity']) for c in d['cases'])
checks['rank2_rank3_close']=all(all(abs(w['rank_results']['2']['reduction_fraction']-w['rank_results']['3']['reduction_fraction'])<=0.02 for w in c['width_sensitivity']) for c in d['cases'])
checks['outer_extension_096_beats_098']=d['outer_extension']['best']['midpoint']==0.96 and next(x for x in d['outer_extension']['candidates'] if x['midpoint']==0.98)['reduction_fraction']<0.5
checks['resolution_step_angle']=d['max_step_angle_deg']<0.01
# double preload: normalization reproduces 0.05G envelope and finite response remains O(1) SD
p=ROOT/'preload/double_path_null_control/results/R196_DOUBLE_PRELOAD_NULL_CONTROL.json'
d=json.loads(p.read_text())
checks['double_preload_coeff_ratio']=abs(d['coefficient_ratio_double_over_single']-88.0372787160093)<1e-9
checks['double_preload_norm']=abs(d['double_zero']['profile_norm_SD']-3.164130601216395)<1e-9
checks['double_preload_not_eliminated']=d['double_zero']['profile_norm_SD']>1.0
checks['double_preload_step_angles']=max(d['double_zero']['step_angle_deg'].values())<0.01
status='PASS' if all(checks.values()) else 'FAIL'
out={'status':status,'checks':checks}
(ROOT/'results/round17_new_controls_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
raise SystemExit(0 if status=='PASS' else 1)
