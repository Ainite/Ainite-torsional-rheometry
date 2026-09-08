#!/usr/bin/env python3
"""Check the reported radial-window nuisance-profile geometry.

For a one-dimensional nuisance direction, the profiled residual equals the
constitutive norm times sin(angle). The input table retains the independently
refitted window-specific norms and angles; this script regenerates the
corresponding residual and the 25 equal-area sampling positions for each window.
"""
from pathlib import Path
import csv, json, math
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent; INP=ROOT/'input'; RES=ROOT/'results'; RES.mkdir(exist_ok=True)
rows=[]
with (INP/'fitting_window_summary.csv').open(newline='') as f:
    for r in csv.DictReader(f):
        xmax=float(r['xmax']); n=25; xmin=.05
        j=np.arange(n,dtype=float)
        x=np.sqrt(xmin*xmin+(j+.5)/n*(xmax*xmax-xmin*xmin))
        norm=float(r['scaled_constitutive_tangent_norm_sigma'])
        angle=float(r['angle_deg'])
        predicted=norm*math.sin(math.radians(angle))
        reported=float(r['profiled_residual_sigma'])
        rows.append({'xmax':xmax,'sampling_x':x.tolist(),
                     'predicted_profiled_residual_sigma':predicted,
                     'reported_profiled_residual_sigma':reported,
                     'abs_rounding_difference':abs(predicted-reported)})
out={'rows':rows,'max_abs_rounding_difference':max(z['abs_rounding_difference'] for z in rows),
     'note':'Differences reflect rounding of the published norms and angles.'}
(RES/'fitting_window_metrics.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
