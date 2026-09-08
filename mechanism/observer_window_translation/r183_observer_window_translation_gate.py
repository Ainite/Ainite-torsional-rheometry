#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np
import scipy.sparse.linalg as spla
import r183_core as C
HERE=Path(__file__).resolve().parent; INP=HERE/'input'; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
RULE={
 'schema':'U1-R183-observer-window-translation-preregister-v1',
 'state':'fixed 34x20 aspect=5 edge_width=0.10 baseline from R181',
 'observer_xmax':[0.75,0.85,0.95],
 'observer_points':'25 equal-area samples from x=0.05 to each xmax',
 'smooth_halfwidth':0.02,
 'source_pair_relative_to_observer_xmax':[-0.03,+0.03],
 'decision_cases':[0.75,0.95],
 'control_case':0.85,
 'support_gate':'BOTH shifted observer cutoffs (0.75 and 0.95) must have angle >150 deg between existing and added-band components in the orthogonal complement of that observer-specific rank-3 interface span, and >=50% reduction of orthogonal norm after adding the band',
 'control_gate':'xmax=0.85 must reproduce angle >150 deg and >=50% reduction',
 'validity':'all pressure-tangent centered step-angle checks <0.01 deg and frozen baseline equilibrium residual infinity norms <1e-7',
 'claim_boundary':'A pass supports observer-window translation on one fixed FE state and interface law. It would not by itself prove a Saint-Venant theorem or universality across loading/material families.'
}
(OUT/'R183_PREREGISTERED_RULE.json').write_text(json.dumps(RULE,indent=2)+'\n'); print('PREREGISTERED',json.dumps(RULE,indent=2),flush=True)

def xs_for(xmax):
    j=np.arange(25.); return np.sqrt(.05**2+(j+.5)/25*(xmax**2-.05**2))

def samp(x,v,xs):
    o=np.argsort(x); return np.interp(xs,np.asarray(x)[o],np.asarray(v)[o])

def tangent_custom(mesh,yb,ys,dyb,dys,xs,direct_c=0.,center=None,halfwidth=.02):
    vals={}
    for h in [1e-4,5e-5]:
        xp,pp,ap=C.raw_prof(mesh,yb+h*dyb,ys+h*dys,+h*direct_c,center,halfwidth)
        xm,pm,am=C.raw_prof(mesh,yb-h*dyb,ys-h*dys,-h*direct_c,center,halfwidth)
        vals[h]=((samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h),(ap-am)/(2*h))
    return vals[5e-5][0], C.angle(vals[1e-4][0],vals[5e-5][0]), float(vals[5e-5][1])

def projection(g,U):
    r=g-U@(U.T@g); n=np.linalg.norm(g)
    return {'norm':float(n),'orth_norm':float(np.linalg.norm(r)),'orth_fraction':float(np.linalg.norm(r)/n),'angle_deg':float(math.degrees(math.asin(np.clip(np.linalg.norm(r)/n,0,1))))}

mesh=C.m.build(C.NR,C.NZ,5.0,.10,C.EDGE_CELLS,None,'tapered_springs')
z=np.load(INP/'baseline_edgewidth_0.10_aspect5_34x20.npz'); yb=np.asarray(z['yb'],float); ys=np.asarray(z['ys'],float)
_,gb,Hb=C.m.assemble(mesh,yb,C.K,0.,C.EPS,True,*C.BETA); _,gs,Hs=C.m.assemble(mesh,ys,C.K,C.GAM,C.EPS,True,*C.BETA)
# Observer-independent state derivatives for interface coordinates.
beta_dys=[]
for ch in range(3):
    sb=C.beta_residual_derivative(mesh,yb,0.,ch); ss=C.beta_residual_derivative(mesh,ys,C.GAM,ch)
    beta_dys.append((spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss)))
# Observer-independent source derivatives at all six preregistered centers.
centers=sorted(set(round(x+d,10) for x in RULE['observer_xmax'] for d in RULE['source_pair_relative_to_observer_xmax']))
src_dys={}
for center in centers:
    mask=C.smooth_inner_weight(np.asarray(mesh['Rq']),center,.02)
    sb=C.assemble_source(mesh,yb,mask,True); ss=C.assemble_source(mesh,ys,mask,False)
    src_dys[center]=(spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss),float(np.sum(mask*np.asarray(mesh['W']))/np.sum(np.asarray(mesh['W']))))
rows=[]; maxstep=0.
for xmax in RULE['observer_xmax']:
    xs=xs_for(xmax); cols=[]; bchecks=[]
    for ch,(dyb,dys) in enumerate(beta_dys):
        g,sa,rd=tangent_custom(mesh,yb,ys,dyb,dys,xs,0.,None,.02); cols.append(g); bchecks.append({'channel':ch,'step_angle_deg':sa}); maxstep=max(maxstep,sa)
    B=np.column_stack(cols); U,sv,_=np.linalg.svd(B,full_matrices=False); orth=lambda g:g-U@(U.T@g)
    prof={}; wrows=[]
    for rel in RULE['source_pair_relative_to_observer_xmax']:
        center=round(xmax+rel,10); dyb,dys,mass=src_dys[center]
        g,sa,rd=tangent_custom(mesh,yb,ys,dyb,dys,xs,1.,center,.02); prof[rel]=g
        wrows.append({'relative_center':rel,'center':center,**projection(g,U),'step_angle_deg':sa,'mass_fraction':mass}); maxstep=max(maxstep,sa)
    a=prof[-.03]; ex=prof[.03]; band=ex-a; oa=orth(a); ob=orth(band); os=orth(ex)
    ang=C.angle(oa,ob); red=1-float(np.linalg.norm(os)/np.linalg.norm(oa)); passed=bool(ang>150 and red>=.5)
    row={'xmax':xmax,'interface_singular_values':sv.tolist(),'interface_step_checks':bchecks,'windows':wrows,'cancellation':{'pair_absolute':[xmax-.03,xmax+.03],'orth_angle_deg':ang,'orth_norm_before':float(np.linalg.norm(oa)),'orth_norm_added_band':float(np.linalg.norm(ob)),'orth_norm_after':float(np.linalg.norm(os)),'reduction_fraction':red,'pass':passed}}
    rows.append(row)
    np.savetxt(OUT/f'profiles_xmax_{xmax:.2f}.csv',np.column_stack([xs,cols[0],cols[1],cols[2],prof[-.03],prof[.03]]),delimiter=',',header='x,beta_r_log,beta_theta_log,beta_z_log,inner,expanded',comments='')
    print(json.dumps({'xmax':xmax,'cancellation':row['cancellation'],'sv':sv.tolist()},indent=2),flush=True)
valid=bool(maxstep<.01 and np.linalg.norm(gb,np.inf)<1e-7 and np.linalg.norm(gs,np.inf)<1e-7)
by={r['xmax']:r for r in rows}; shifted=all(by[x]['cancellation']['pass'] for x in RULE['decision_cases']); control=by[.85]['cancellation']['pass']
if not valid: verdict='INVALID_NUMERICAL_GATE'
elif shifted and control: verdict='OBSERVER_WINDOW_TRANSLATION_SUPPORTED'
elif control: verdict='OBSERVER_WINDOW_TRANSLATION_NOT_SUPPORTED__CONTROL_REPRODUCED'
else: verdict='OBSERVER_WINDOW_GATE_FAILED_CONTROL'
out={'schema':'U1-R183-observer-window-translation-v1','preregistered_rule':RULE,'baseline_grad_inf_pre':float(np.linalg.norm(gb,np.inf)),'baseline_grad_inf_twist':float(np.linalg.norm(gs,np.inf)),'max_step_angle_deg':maxstep,'valid':valid,'cases':rows,'verdict':verdict}
(OUT/'R183_OBSERVER_WINDOW_TRANSLATION_GATE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'verdict':verdict,'valid':valid,'cases':[{'xmax':r['xmax'],**r['cancellation']} for r in rows]},indent=2),flush=True)
