#!/usr/bin/env python3
from pathlib import Path
import json, math, numpy as np
import scipy.sparse.linalg as spla
import r184_core as C
HERE=Path(__file__).resolve().parent; INP=HERE/'input'; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
RULE={
 'schema':'U1-R184-nested-cancellation-preregister-v1',
 'state':'fixed R181 34x20 aspect=5 edge_width=0.10 equilibrium state',
 'new_observer_xmax':[0.70,0.80,0.90],
 'samples':'25 equal-area pressure points from 0.05 to xmax',
 'source_pair':'C2 halfwidth=0.02, centers xmax-0.03 and xmax+0.03',
 'level1_direct_state_gate':'for EACH new cutoff: direct-vs-state added-band angle >160 deg; direct/state orthogonal-norm ratio in [0.75,1.33]; total added-band orthogonal norm <=0.35*max(direct,state)',
 'level2_interior_gate':'for EACH new cutoff: total added-band orthogonal component angle >150 deg to the pre-existing inner orthogonal component, and adding it reduces the inner orthogonal norm by >=50%',
 'validity':'all centered finite-difference step-angle checks <0.01 deg; decomposition closure relative error <1e-5; baseline equilibrium residual infinity norms <1e-7',
 'verdict_rule':'all three new cutoffs pass both levels -> NESTED_OBSERVER_BOUNDARY_CANCELLATION_SUPPORTED',
 'claim_boundary':'This is a finite-BVP mechanism on one material/loading/geometry family. It does not establish a general Saint-Venant theorem.'
}
(OUT/'R184_PREREGISTERED_RULE.json').write_text(json.dumps(RULE,indent=2)+'\n'); print('PREREGISTERED',json.dumps(RULE,indent=2),flush=True)

def xs_for(xmax):
    j=np.arange(25.); return np.sqrt(.05**2+(j+.5)/25*(xmax*xmax-.05**2))
def samp(x,v,xs):
    o=np.argsort(x); return np.interp(xs,np.asarray(x)[o],np.asarray(v)[o])
def ang(a,b): return C.angle(a,b)
def prof_deriv(mesh,yb,ys,dyb,dys,xs,center,kind):
    vals={}
    for h in [1e-4,5e-5]:
        if kind=='full': plus=(yb+h*dyb,ys+h*dys,+h); minus=(yb-h*dyb,ys-h*dys,-h)
        elif kind=='direct': plus=(yb,ys,+h); minus=(yb,ys,-h)
        elif kind=='state': plus=(yb+h*dyb,ys+h*dys,0.); minus=(yb-h*dyb,ys-h*dys,0.)
        else: raise ValueError(kind)
        xp,pp,_=C.raw_prof(mesh,plus[0],plus[1],plus[2],center,.02)
        xm,pm,_=C.raw_prof(mesh,minus[0],minus[1],minus[2],center,.02)
        vals[h]=(samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h)
    return vals[5e-5], ang(vals[1e-4],vals[5e-5])
def beta_profile(mesh,yb,ys,db,ds,xs):
    vals={}
    for h in [1e-4,5e-5]:
        xp,pp,_=C.raw_prof(mesh,yb+h*db,ys+h*ds,0.,None,.02)
        xm,pm,_=C.raw_prof(mesh,yb-h*db,ys-h*ds,0.,None,.02)
        vals[h]=(samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h)
    return vals[5e-5], ang(vals[1e-4],vals[5e-5])

mesh=C.m.build(C.NR,C.NZ,5.,.1,C.EDGE_CELLS,None,'tapered_springs')
z=np.load(INP/'baseline_edgewidth_0.10_aspect5_34x20.npz'); yb=np.asarray(z['yb']); ys=np.asarray(z['ys'])
_,gb,Hb=C.m.assemble(mesh,yb,C.K,0.,C.EPS,True,*C.BETA); _,gs,Hs=C.m.assemble(mesh,ys,C.K,C.GAM,C.EPS,True,*C.BETA)
beta_dys=[]
for ch in range(3):
    sb=C.beta_residual_derivative(mesh,yb,0.,ch); ss=C.beta_residual_derivative(mesh,ys,C.GAM,ch)
    beta_dys.append((spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss)))
centers=sorted(set(round(x+d,10) for x in RULE['new_observer_xmax'] for d in [-.03,.03])); src_dys={}
for c in centers:
    mask=C.smooth_inner_weight(np.asarray(mesh['Rq']),c,.02); sb=C.assemble_source(mesh,yb,mask,True); ss=C.assemble_source(mesh,ys,mask,False)
    src_dys[c]=(spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss))
rows=[]; maxstep=0.; maxclosure=0.
for xmax in RULE['new_observer_xmax']:
    xs=xs_for(xmax); cols=[]
    for db,ds in beta_dys:
        g,sa=beta_profile(mesh,yb,ys,db,ds,xs); cols.append(g); maxstep=max(maxstep,sa)
    U,sv,_=np.linalg.svd(np.column_stack(cols),full_matrices=False); orth=lambda g:g-U@(U.T@g)
    data={}
    for center in [round(xmax-.03,10),round(xmax+.03,10)]:
        db,ds=src_dys[center]; data[center]={}
        for kind in ['full','direct','state']:
            g,sa=prof_deriv(mesh,yb,ys,db,ds,xs,center,kind); data[center][kind]=g; maxstep=max(maxstep,sa)
    a=round(xmax-.03,10); e=round(xmax+.03,10); inner=orth(data[a]['full'])
    bands={k:orth(data[e][k]-data[a][k]) for k in ['full','direct','state']}
    nd=np.linalg.norm(bands['direct']); ns=np.linalg.norm(bands['state']); nf=np.linalg.norm(bands['full']); ninner=np.linalg.norm(inner); after=np.linalg.norm(inner+bands['full'])
    ratio=float(nd/ns); dsang=ang(bands['direct'],bands['state']); resid_ratio=float(nf/max(nd,ns)); closure=float(np.linalg.norm(bands['full']-bands['direct']-bands['state'])/max(nf,1e-30)); maxclosure=max(maxclosure,closure)
    level1=bool(dsang>160 and .75<=ratio<=1.33 and resid_ratio<=.35)
    totalang=ang(inner,bands['full']); reduction=1-float(after/ninner); level2=bool(totalang>150 and reduction>=.5)
    row={'xmax':xmax,'pair':[a,e],'interface_singular_values':sv.tolist(),'level1':{'direct_state_angle_deg':dsang,'direct_orth_norm':float(nd),'state_orth_norm':float(ns),'direct_over_state':ratio,'total_band_orth_norm':float(nf),'total_over_max_component':resid_ratio,'closure_rel':closure,'pass':level1},'level2':{'inner_orth_norm':float(ninner),'total_band_angle_to_inner_deg':totalang,'orth_norm_after':float(after),'reduction_fraction':reduction,'pass':level2},'pass':bool(level1 and level2)}
    rows.append(row); print(json.dumps(row,indent=2),flush=True)
valid=bool(maxstep<.01 and maxclosure<1e-5 and np.linalg.norm(gb,np.inf)<1e-7 and np.linalg.norm(gs,np.inf)<1e-7)
if not valid: verdict='INVALID_NUMERICAL_GATE'
elif all(r['pass'] for r in rows): verdict='NESTED_OBSERVER_BOUNDARY_CANCELLATION_SUPPORTED'
else: verdict='NESTED_CANCELLATION_NOT_FULLY_REPLICATED'
out={'schema':'U1-R184-nested-cancellation-v1','preregistered_rule':RULE,'baseline_grad_inf_pre':float(np.linalg.norm(gb,np.inf)),'baseline_grad_inf_twist':float(np.linalg.norm(gs,np.inf)),'max_step_angle_deg':maxstep,'max_decomposition_closure_rel':maxclosure,'valid':valid,'cases':rows,'verdict':verdict}
(OUT/'R184_NESTED_CANCELLATION_GATE.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'verdict':verdict,'valid':valid,'passes':[r['pass'] for r in rows]},indent=2),flush=True)
