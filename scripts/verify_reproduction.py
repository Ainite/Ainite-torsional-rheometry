#!/usr/bin/env python3
"""Verify regenerated observation-space quantities against the deposited references."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent; RES=ROOT/'results'; REF=ROOT/'reference'
finite=json.loads((RES/'finite_state_results.json').read_text())
r025=json.loads((REF/'finite_constitutive_vs_interface_025.json').read_text())
r032=json.loads((REF/'finite_constitutive_vs_interface_032.json').read_text())
sub=json.loads((RES/'subspace_geometry.json').read_text())
subref=json.loads((REF/'subspace_geometry.json').read_text())
cov=json.loads((RES/'observation_covariance.json').read_text())
win=json.loads((RES/'fitting_window_metrics.json').read_text())
checks=[]
def ck(name,a,b,tol=1e-10):
    e=abs(float(a)-float(b)); checks.append({'name':name,'abs_error':e,'tolerance':tol,'pass':e<=tol}); return e<=tol
m=finite['interface_025']
ck('025 residual',m['direct_residual_sigma'],r025['fixed_parameter_residual_sigma'])
ck('025 angle',m['direction_angle_deg'],r025['angle_deg'])
ck('025 profiled',m['direction_profiled_residual_sigma'],r025['direction_profiled_residual_sigma'])
ck('025 radius mismatch',m['radius_mismatch_um'],r025['finite_vs_combined_radius_mismatch_um'])
m=finite['interface_032']
ck('032 residual',m['direct_residual_sigma'],r032['residual_sigma'])
ck('032 angle',m['direction_angle_deg'],r032['angle_deg'])
ck('032 profiled',m['direction_profiled_residual_sigma'],r032['profiled_residual_sigma'])
ck('032 radius mismatch',m['radius_mismatch_um'],r032['radius_mismatch_um'])
for k in ['outside_quadratic_SD','outside_rank2_SD','outside_full_span_SD','finite_constitutive_norm_SD']:
    ck('subspace '+k,sub[k],subref[k])
for i,(a,b) in enumerate(zip(sub['principal_angles_deg'],subref['principal_angles_deg'])):
    ck(f'principal angle {i+1}',a,b)
# Primary covariance anchors printed in the manuscript/OR.
ck('corr_AB',cov['corr_AB'],-0.86769,5e-6)
ck('sigma_AplusB_Pa',cov['sigma_AplusB_Pa'],1.291364,5e-7)
ck('sigma_qP_kPa',cov['sigma_qP_kPa'],0.032284096,5e-10)
ck('sigma_alpha1P_kPa',cov['sigma_alpha1P_kPa'],0.030294409,5e-10)
window_pass=float(win['max_abs_rounding_difference']) < 0.003
checks.append({'name':'fitting-window residual identity (published rounding)','abs_error':float(win['max_abs_rounding_difference']),'tolerance':0.003,'pass':window_pass})
passed=all(c['pass'] for c in checks)
out={'status':'PASS' if passed else 'FAIL','checks':checks}
(RES/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
raise SystemExit(0 if passed else 1)
