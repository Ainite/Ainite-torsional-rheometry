#!/usr/bin/env python3
from pathlib import Path
import json, math
ROOT=Path(__file__).resolve().parent.parent
m=json.loads((ROOT/'meshcheck_30x40/meshcheck_30x40.json').read_text())
r=json.loads((ROOT/'results/global_reactions_38x40.json').read_text())
s=json.loads((ROOT/'results/review17_closure_metrics.json').read_text())['round17_closure']
checks={
 'mesh_025_direct': abs(m['025']['direct_residual_sigma']-0.9878167108482898)<1e-10,
 'mesh_025_profiled': abs(m['025']['direction_profiled_residual_sigma']-0.14456228413155062)<1e-10,
 'mesh_032_direct': abs(m['032']['direct_residual_sigma']-0.1434372629647639)<1e-10,
 'mesh_032_profiled': abs(m['032']['direction_profiled_residual_sigma']-0.14338666145555443)<1e-10,
 'mesh_032_radius': abs(m['032']['radius_mismatch_um']-4.949466621011478)<1e-9,
 'torque_difference': abs(r['constitutive_vs_interface_032']['torque_Nm_difference_interface_minus_constitutive']-5.402127139075841e-05)<1e-12,
 'axial_difference': abs(r['constitutive_vs_interface_032']['axial_force_N_difference_interface_minus_constitutive']+0.434583700641511)<1e-10,
 'tangent_032_direct': abs(s['tangent_pre_solve_predictions']['0.32']['direct_residual_sigma']-0.20842136874723183)<1e-12,
 'tangent_032_profiled': abs(s['tangent_pre_solve_predictions']['0.32']['profiled_residual_sigma']-0.162809134312482)<1e-12,
 'MR_partition': abs(s['baseline_MR_partition_f']-0.2893401015228426)<1e-14,
}
out={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks}
(ROOT/'results/review17_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
raise SystemExit(0 if all(checks.values()) else 1)
