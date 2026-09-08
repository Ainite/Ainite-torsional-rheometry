#!/usr/bin/env python3
from pathlib import Path
import json, math
p=Path(__file__).resolve().parent/'results/R186_MATERIAL_BASELINE_GATE.json'; d=json.loads(p.read_text()); errs=[]
if d['verdict'] not in {'MATERIAL_BASELINE_NESTED_CANCELLATION_REPLICATED','MATERIAL_BASELINE_REPLICATION_FAILED','INVALID_NUMERICAL_GATE'}: errs.append('bad verdict')
if [round(x['f'],2) for x in d['cases']] != [0.15,0.45]: errs.append('wrong material cases')
for c in d['cases']:
 if c['max_step_angle_deg']>=0.01: errs.append(f"step angle f={c['f']}")
 if c['level1']['closure_rel']>=1e-5: errs.append(f"closure f={c['f']}")
 if c['equilibrium_grad_inf_pre']>=1e-7 or c['equilibrium_grad_inf_twist']>=1e-7: errs.append(f"equilibrium f={c['f']}")
 expected1=(c['level1']['direct_state_angle_deg']>160 and .75<=c['level1']['direct_over_state']<=1.33 and c['level1']['total_over_max_component']<=.35)
 expected2=(c['level2']['total_band_angle_to_inner_deg']>150 and c['level2']['reduction_fraction']>=.5)
 if c['level1']['pass'] != expected1: errs.append(f"level1 flag f={c['f']}")
 if c['level2']['pass'] != expected2: errs.append(f"level2 flag f={c['f']}")
 if c['pass'] != (c['valid'] and expected1 and expected2): errs.append(f"case pass f={c['f']}")
print('PASS: R186 verification' if not errs else 'FAIL: '+'; '.join(errs))
raise SystemExit(1 if errs else 0)
