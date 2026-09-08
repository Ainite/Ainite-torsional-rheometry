#!/usr/bin/env python3
"""Recompute the finite-vs-finite pressure-space metrics from supplied nonlinear profiles.

The supplied profiles are the 25 equal-area observations extracted from the fully
converged 38x40 finite-element states. This script independently recomputes the
norms, direction angles, scalar-profiled residuals, and quadratic-fit increments
used in the manuscript.
"""
from pathlib import Path
import json, math
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent; INP=ROOT/'input'; RES=ROOT/'results'; REF=ROOT/'reference'
RES.mkdir(exist_ok=True)
SIGMA=2.0

def load_profile(path):
    d=np.genfromtxt(path,delimiter=',',names=True)
    names=d.dtype.names
    x=np.asarray(d['x'],float)
    c=np.asarray(d['finite_constitutive_shift_Pa'],float)
    i=np.asarray(d[names[2]],float)
    return x,c,i

def metrics(x,c,i):
    nc=float(np.linalg.norm(c)/SIGMA); ni=float(np.linalg.norm(i)/SIGMA)
    residual=float(np.linalg.norm(c-i)/SIGMA)
    cos=float(np.dot(c,i)/(np.linalg.norm(c)*np.linalg.norm(i)))
    angle=float(math.degrees(math.acos(np.clip(cos,-1.,1.))))
    eta=float(np.dot(c,i)/np.dot(i,i))
    prof=float(np.linalg.norm(c-eta*i)/SIGMA)
    X=np.column_stack([np.ones_like(x),x*x])
    return dict(constitutive_norm_sigma=nc,interface_norm_sigma=ni,
                direct_residual_sigma=residual,direction_angle_deg=angle,
                optimal_scalar=eta,direction_profiled_residual_sigma=prof,
                constitutive_dAB_Pa=np.linalg.lstsq(X,c,rcond=None)[0].tolist(),
                interface_dAB_Pa=np.linalg.lstsq(X,i,rcond=None)[0].tolist())

x25,c25,i25=load_profile(INP/'finite_profile_reference.csv')
x32,c32,i32=load_profile(INP/'finite_profile_032.csv')
if not np.array_equal(x25,x32) or not np.array_equal(c25,c32):
    raise RuntimeError('0.25 and 0.32 profile files do not share the same constitutive observation vector')
r25=json.loads((REF/'finite_constitutive_vs_interface_025.json').read_text())
r32=json.loads((REF/'finite_constitutive_vs_interface_032.json').read_text())
m25=metrics(x25,c25,i25); m32=metrics(x32,c32,i32)
m25.update(radius_mismatch_um=float(r25['finite_vs_combined_radius_mismatch_um']))
m32.update(radius_mismatch_um=float(r32['radius_mismatch_um']))
out={'interface_025':m25,'interface_032':m32}
(RES/'finite_state_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
