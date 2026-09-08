#!/usr/bin/env python3
from pathlib import Path
import json, math
ROOT=Path(__file__).resolve().parent.parent
checks={}
svd=json.loads((ROOT/'mechanism/band_operator_svd/results/band_operator_svd.json').read_text())
expected={'0.73':(0.8970811902581886,2.016863461953154,0.9982107792967363),
          '0.83':(0.7023859502399326,4.5092291672273115,0.995099274104543),
          '0.93':(0.48731651617253025,7.669735616852986,0.9574340856140202)}
for k,(e2,er,r2) in expected.items():
    c=svd['cases'][k]
    checks[f'svd_E2_{k}']=abs(c['first2_operator_energy_fraction']-e2)<1e-10
    checks[f'svd_erank_{k}']=abs(c['entropy_effective_rank']-er)<1e-10
    checks[f'svd_fullresp_{k}']=abs(c['full_Q1_response_fraction_in_first2_left_modes']-r2)<1e-10
checks['svd_not_uniform_rank2']=svd['cases']['0.93']['first2_operator_energy_fraction']<0.5 and svd['cases']['0.83']['entropy_effective_rank']>4
checks['source_weight_output_compression']=all(svd['cases'][k]['full_Q1_response_fraction_in_first2_left_modes']>0.95 for k in expected)
inner=json.loads((ROOT/'mechanism/interior_support_control/results/interior_support_control.json').read_text())
checks['inner_support_verdict']=inner.get('verdict')=='INNER_SUPPORT_EFFECT_SUPPORTED'
for k,c in inner['cases'].items():
    checks[f'inner_support_{k}']=c['support_effect_pass'] and c['inner60']['rank3_orth_fraction']>0.5 and c['ratio_inner_to_full']>15
status='PASS' if all(checks.values()) else 'FAIL'
out={'status':status,'checks':checks}
(ROOT/'results/review17_mechanism_closure_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
raise SystemExit(0 if status=='PASS' else 1)
