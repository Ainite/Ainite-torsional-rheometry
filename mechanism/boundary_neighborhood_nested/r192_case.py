#!/usr/bin/env python3
import argparse,json,numpy as np
from pathlib import Path
import scipy.sparse.linalg as spla
import r190_core as C
HERE=Path(__file__).resolve().parent; INP=HERE/'input'; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
def xs_for(xmax):
 j=np.arange(25.); return np.sqrt(.05**2+(j+.5)/25*(xmax*xmax-.05**2))
def samp(x,v,xs):
 o=np.argsort(x); return np.interp(xs,np.asarray(x)[o],np.asarray(v)[o])
def prof(mesh,yb,ys,dyb,dys,xs,center,kind):
 vals={}
 for h in [1e-4,5e-5]:
  if kind=='full': plus=(yb+h*dyb,ys+h*dys,+h); minus=(yb-h*dyb,ys-h*dys,-h)
  elif kind=='direct': plus=(yb,ys,+h); minus=(yb,ys,-h)
  elif kind=='state': plus=(yb+h*dyb,ys+h*dys,0.); minus=(yb-h*dyb,ys-h*dys,0.)
  xp,pp,_=C.raw_prof(mesh,plus[0],plus[1],plus[2],center,.02); xm,pm,_=C.raw_prof(mesh,minus[0],minus[1],minus[2],center,.02)
  vals[h]=(samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h)
 return vals[5e-5],C.angle(vals[1e-4],vals[5e-5])
def beta_profile(mesh,yb,ys,db,ds,xs):
 vals={}
 for h in [1e-4,5e-5]:
  xp,pp,_=C.raw_prof(mesh,yb+h*db,ys+h*ds,0.,None,.02); xm,pm,_=C.raw_prof(mesh,yb-h*db,ys-h*ds,0.,None,.02)
  vals[h]=(samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h)
 return vals[5e-5],C.angle(vals[1e-4],vals[5e-5])
ap=argparse.ArgumentParser();ap.add_argument('--xmax',type=float,required=True);a=ap.parse_args();xmax=float(a.xmax)
grid=np.round(np.arange(.60,.9601,.02),10); mids=sorted(grid,key=lambda q:(abs(float(q)-xmax),float(q)))[:2]; mids=sorted(float(q) for q in mids)
mesh=C.m.build(C.NR,C.NZ,5.,.1,C.EDGE_CELLS,None,'tapered_springs');z=np.load(INP/'baseline_edgewidth_0.10_aspect5_34x20.npz');yb=np.asarray(z['yb']);ys=np.asarray(z['ys'])
_,gb,Hb=C.m.assemble(mesh,yb,C.K,0.,C.EPS,True,*C.BETA);_,gs,Hs=C.m.assemble(mesh,ys,C.K,C.GAM,C.EPS,True,*C.BETA)
beta=[]
for ch in range(3):
 sb=C.beta_residual_derivative(mesh,yb,0.,ch);ss=C.beta_residual_derivative(mesh,ys,C.GAM,ch);beta.append((spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss)))
xs=xs_for(xmax); cols=[]; maxstep=0.
for db,ds in beta:
 g,sa=beta_profile(mesh,yb,ys,db,ds,xs);cols.append(g);maxstep=max(maxstep,sa)
U,sv,_=np.linalg.svd(np.column_stack(cols),full_matrices=False);orth=lambda g:g-U@(U.T@g)
# unmasked full source for scale floor
ones=np.ones_like(np.asarray(mesh['Rq']),float);sb=C.assemble_source(mesh,yb,ones,True);ss=C.assemble_source(mesh,ys,ones,False);fdy=(spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss));gf,sa=prof(mesh,yb,ys,fdy[0],fdy[1],xs,None,'full');maxstep=max(maxstep,sa);fullorth=float(np.linalg.norm(orth(gf)));floor=.10*fullorth
centers=sorted(set(round(m+d,10) for m in mids for d in (-.03,.03))); src={}
for c in centers:
 mask=C.smooth_inner_weight(np.asarray(mesh['Rq']),c,.02);sb=C.assemble_source(mesh,yb,mask,True);ss=C.assemble_source(mesh,ys,mask,False);db=spla.spsolve(Hb,-sb);ds=spla.spsolve(Hs,-ss);src[c]={'dy':(db,ds)}
 for kind in ('full','direct','state'):
  g,sa=prof(mesh,yb,ys,db,ds,xs,c,kind);src[c][kind]=g;maxstep=max(maxstep,sa)
recs=[]
for mid in mids:
 lo=round(mid-.03,10);hi=round(mid+.03,10)
 inner=orth(src[lo]['full']);band=orth(src[hi]['full']-src[lo]['full']);ni=float(np.linalg.norm(inner));nb=float(np.linalg.norm(band));after=float(np.linalg.norm(inner+band));fa=C.angle(inner,band);red=1-after/max(ni,1e-30);fullpass=bool(ni>=floor and nb>=floor and fa>150 and red>=.5)
 bd=orth(src[hi]['direct']-src[lo]['direct']);bs=orth(src[hi]['state']-src[lo]['state']);bf=band;nd=float(np.linalg.norm(bd));ns=float(np.linalg.norm(bs));nf=float(np.linalg.norm(bf));ratio=nd/max(ns,1e-30);dsa=C.angle(bd,bs);rr=nf/max(nd,ns,1e-30);cl=float(np.linalg.norm(bf-bd-bs)/max(nd,ns,1e-30));mechpass=bool(dsa>160 and .75<=ratio<=1.33 and rr<=.35 and cl<1e-5)
 recs.append({'midpoint':mid,'distance_to_xmax':abs(mid-xmax),'pair':[lo,hi],'inner_orth_norm':ni,'band_orth_norm':nb,'full_angle_deg':fa,'reduction_fraction':red,'full_gate_pass':fullpass,'direct_state_angle_deg':dsa,'direct_over_state':ratio,'full_over_max_component':rr,'closure_over_max_component':cl,'mechanics_pass':mechpass,'band_pass':bool(fullpass and mechpass)})
numerical=bool(maxstep<.01 and np.linalg.norm(gb,np.inf)<1e-7 and np.linalg.norm(gs,np.inf)<1e-7); neighborhood=any(r['band_pass'] for r in recs);casepass=bool(numerical and neighborhood)
out={'xmax':xmax,'preselected_midpoints':mids,'interface_singular_values':sv.tolist(),'full_orth_norm':fullorth,'scan_floor':floor,'bands':recs,'boundary_neighborhood_pass':neighborhood,'max_step_angle_deg':maxstep,'baseline_grad_inf_pre':float(np.linalg.norm(gb,np.inf)),'baseline_grad_inf_twist':float(np.linalg.norm(gs,np.inf)),'numerical_valid':numerical,'pass':casepass}
p=OUT/f'R192_CASE_XMAX_{xmax:.2f}.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
