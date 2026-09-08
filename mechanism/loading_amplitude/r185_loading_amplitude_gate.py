#!/usr/bin/env python3
from pathlib import Path
import json, math, numpy as np
import scipy.sparse.linalg as spla
import r185_core as C
HERE=Path(__file__).resolve().parent; INP=HERE/'input'; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
RULE={
 'schema':'U1-R185-loading-amplitude-preregister-v1',
 'state_family':'same 34x20 aspect=5 edge_width=0.10 material/preload/interface; only gamma_R changes',
 'new_gamma_R':[0.15,0.25],
 'observer_xmax':0.80,
 'source_pair':[0.77,0.83],
 'smooth_halfwidth':0.02,
 'level1':'for EACH gamma: direct-vs-state added-band angle >160 deg; direct/state orthogonal norm ratio in [0.75,1.33]; total added-band orthogonal norm <=0.35*max(direct,state)',
 'level2':'for EACH gamma: total added-band orthogonal component angle >150 deg to pre-existing interior orthogonal component; adding it reduces inner orthogonal norm by >=50%',
 'validity':'all centered step-angle checks <0.01 deg; decomposition closure <1e-5; equilibrium residual infinity norms <1e-7',
 'verdict_rule':'both new gamma cases pass both levels -> LOADING_AMPLITUDE_NESTED_CANCELLATION_REPLICATED',
 'claim_boundary':'A pass establishes loading-amplitude robustness over gamma_R=0.15,0.20,0.25 for this material/preload/geometry family; material-family robustness remains open.'
}
(OUT/'R185_PREREGISTERED_RULE.json').write_text(json.dumps(RULE,indent=2)+'\n'); print('PREREGISTERED',json.dumps(RULE,indent=2),flush=True)

def xs_for(xmax):
 j=np.arange(25.); return np.sqrt(.05**2+(j+.5)/25*(xmax*xmax-.05**2))
def samp(x,v,xs):
 o=np.argsort(x); return np.interp(xs,np.asarray(x)[o],np.asarray(v)[o])
def prof_deriv(mesh,yb,ys,dyb,dys,xs,center,kind):
 vals={}
 for h in [1e-4,5e-5]:
  if kind=='full': plus=(yb+h*dyb,ys+h*dys,+h); minus=(yb-h*dyb,ys-h*dys,-h)
  elif kind=='direct': plus=(yb,ys,+h); minus=(yb,ys,-h)
  elif kind=='state': plus=(yb+h*dyb,ys+h*dys,0.); minus=(yb-h*dyb,ys-h*dys,0.)
  else: raise ValueError(kind)
  xp,pp,_=C.raw_prof(mesh,plus[0],plus[1],plus[2],center,.02); xm,pm,_=C.raw_prof(mesh,minus[0],minus[1],minus[2],center,.02)
  vals[h]=(samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h)
 return vals[5e-5], C.angle(vals[1e-4],vals[5e-5])
def beta_prof(mesh,yb,ys,db,ds,xs):
 vals={}
 for h in [1e-4,5e-5]:
  xp,pp,_=C.raw_prof(mesh,yb+h*db,ys+h*ds,0.,None,.02); xm,pm,_=C.raw_prof(mesh,yb-h*db,ys-h*ds,0.,None,.02)
  vals[h]=(samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h)
 return vals[5e-5], C.angle(vals[1e-4],vals[5e-5])

mesh=C.m.build(C.NR,C.NZ,5.,.1,C.EDGE_CELLS,None,'tapered_springs'); z=np.load(INP/'baseline_edgewidth_0.10_aspect5_34x20.npz'); yb=np.asarray(z['yb']); ys20=np.asarray(z['ys']); xs=xs_for(.80)
# Pre-state equilibrium is common to all gamma.
_,gb,Hb0=C.m.assemble(mesh,yb,C.K,0.,C.EPS,True,*C.BETA)
rows=[]
for gamma in RULE['new_gamma_R']:
 C.GAM=float(gamma)
 # Re-equilibrate from the frozen gamma=0.20 state; for 0.25 use a short continuation if direct Newton fails.
 ys,ok,h=C.m.newton(mesh,ys20.copy(),C.K,gamma,C.EPS,*C.BETA,tol=1e-8,maxit=50)
 if not ok and gamma>.20:
  ytmp=ys20.copy(); hist=[]; ok=True
  for gg in [.225,.25]:
   ytmp,oo,hh=C.m.newton(mesh,ytmp,C.K,gg,C.EPS,*C.BETA,tol=1e-8,maxit=50); hist+=hh; ok &= oo
   if not oo: break
  ys=ytmp; h=hist
 if not ok: raise RuntimeError(f'gamma solve failed {gamma}: {h[-1]}')
 np.savez_compressed(OUT/f'state_gamma_{gamma:.2f}.npz',yb=yb,ys=ys,gamma=gamma)
 _,gs,Hs=C.m.assemble(mesh,ys,C.K,gamma,C.EPS,True,*C.BETA)
 # Interface tangent basis at this gamma.
 beta_dys=[]; cols=[]; maxstep=0.
 for ch in range(3):
  sb=C.beta_residual_derivative(mesh,yb,0.,ch); ss=C.beta_residual_derivative(mesh,ys,gamma,ch); db=spla.spsolve(Hb0,-sb); ds=spla.spsolve(Hs,-ss); beta_dys.append((db,ds)); g,sa=beta_prof(mesh,yb,ys,db,ds,xs); cols.append(g); maxstep=max(maxstep,sa)
 U,sv,_=np.linalg.svd(np.column_stack(cols),full_matrices=False); orth=lambda g:g-U@(U.T@g)
 # Source derivatives at the two observer-boundary centers.
 data={}
 for center in RULE['source_pair']:
  mask=C.smooth_inner_weight(np.asarray(mesh['Rq']),center,.02); sb=C.assemble_source(mesh,yb,mask,True); ss=C.assemble_source(mesh,ys,mask,False); db=spla.spsolve(Hb0,-sb); ds=spla.spsolve(Hs,-ss); data[center]={}
  for kind in ['full','direct','state']:
   g,sa=prof_deriv(mesh,yb,ys,db,ds,xs,center,kind); data[center][kind]=g; maxstep=max(maxstep,sa)
 a,e=RULE['source_pair']; inner=orth(data[a]['full']); bands={k:orth(data[e][k]-data[a][k]) for k in ['full','direct','state']}
 nd=np.linalg.norm(bands['direct']); ns=np.linalg.norm(bands['state']); nf=np.linalg.norm(bands['full']); ni=np.linalg.norm(inner); after=np.linalg.norm(inner+bands['full'])
 dsang=C.angle(bands['direct'],bands['state']); ratio=float(nd/ns); resid_ratio=float(nf/max(nd,ns)); closure=float(np.linalg.norm(bands['full']-bands['direct']-bands['state'])/max(nf,1e-30)); l1=bool(dsang>160 and .75<=ratio<=1.33 and resid_ratio<=.35)
 totalang=C.angle(inner,bands['full']); reduction=1-float(after/ni); l2=bool(totalang>150 and reduction>=.5)
 valid=bool(maxstep<.01 and closure<1e-5 and np.linalg.norm(gb,np.inf)<1e-7 and np.linalg.norm(gs,np.inf)<1e-7)
 rec={'gamma_R':gamma,'equilibrium_grad_inf':float(np.linalg.norm(gs,np.inf)),'interface_singular_values':sv.tolist(),'max_step_angle_deg':maxstep,'level1':{'direct_state_angle_deg':dsang,'direct_orth_norm':float(nd),'state_orth_norm':float(ns),'direct_over_state':ratio,'total_band_orth_norm':float(nf),'total_over_max_component':resid_ratio,'closure_rel':closure,'pass':l1},'level2':{'inner_orth_norm':float(ni),'total_band_angle_to_inner_deg':totalang,'orth_norm_after':float(after),'reduction_fraction':reduction,'pass':l2},'valid':valid,'pass':bool(valid and l1 and l2)}
 rows.append(rec); print(json.dumps(rec,indent=2),flush=True)
if not all(r['valid'] for r in rows): verdict='INVALID_NUMERICAL_GATE'
elif all(r['pass'] for r in rows): verdict='LOADING_AMPLITUDE_NESTED_CANCELLATION_REPLICATED'
else: verdict='LOADING_AMPLITUDE_REPLICATION_FAILED'
out={'schema':'U1-R185-loading-amplitude-v1','preregistered_rule':RULE,'baseline_grad_inf_pre':float(np.linalg.norm(gb,np.inf)),'cases':rows,'verdict':verdict}
(OUT/'R185_LOADING_AMPLITUDE_GATE.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'verdict':verdict,'passes':[r['pass'] for r in rows]},indent=2),flush=True)
