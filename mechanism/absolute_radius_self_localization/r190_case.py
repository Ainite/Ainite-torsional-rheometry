#!/usr/bin/env python3
import argparse, json, numpy as np
from pathlib import Path
import scipy.sparse.linalg as spla
import r190_core as C
HERE=Path(__file__).resolve().parent; INP=HERE/'input'; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)

def xs_for(xmax):
    j=np.arange(25.); return np.sqrt(.05**2+(j+.5)/25*(xmax*xmax-.05**2))
def samp(x,v,xs):
    o=np.argsort(x); return np.interp(xs,np.asarray(x)[o],np.asarray(v)[o])
def ang(a,b): return C.angle(a,b)
def prof(mesh,yb,ys,dyb,dys,xs,center,kind):
    vals={}
    for h in [1e-4,5e-5]:
        if kind=='full': plus=(yb+h*dyb,ys+h*dys,+h); minus=(yb-h*dyb,ys-h*dys,-h)
        elif kind=='direct': plus=(yb,ys,+h); minus=(yb,ys,-h)
        elif kind=='state': plus=(yb+h*dyb,ys+h*dys,0.); minus=(yb-h*dyb,ys-h*dys,0.)
        xp,pp,_=C.raw_prof(mesh,plus[0],plus[1],plus[2],center,.02); xm,pm,_=C.raw_prof(mesh,minus[0],minus[1],minus[2],center,.02)
        vals[h]=(samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h)
    return vals[5e-5],ang(vals[1e-4],vals[5e-5])
def beta_profile(mesh,yb,ys,db,ds,xs):
    vals={}
    for h in [1e-4,5e-5]:
        xp,pp,_=C.raw_prof(mesh,yb+h*db,ys+h*ds,0.,None,.02); xm,pm,_=C.raw_prof(mesh,yb-h*db,ys-h*ds,0.,None,.02)
        vals[h]=(samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h)
    return vals[5e-5],ang(vals[1e-4],vals[5e-5])

ap=argparse.ArgumentParser(); ap.add_argument('--xmax',type=float,required=True); args=ap.parse_args(); xmax=float(args.xmax)
mesh=C.m.build(C.NR,C.NZ,5.,.1,C.EDGE_CELLS,None,'tapered_springs'); z=np.load(INP/'baseline_edgewidth_0.10_aspect5_34x20.npz'); yb=np.asarray(z['yb']); ys=np.asarray(z['ys'])
_,gb,Hb=C.m.assemble(mesh,yb,C.K,0.,C.EPS,True,*C.BETA); _,gs,Hs=C.m.assemble(mesh,ys,C.K,C.GAM,C.EPS,True,*C.BETA)
beta_dys=[]
for ch in range(3):
    sb=C.beta_residual_derivative(mesh,yb,0.,ch); ss=C.beta_residual_derivative(mesh,ys,C.GAM,ch); beta_dys.append((spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss)))
midpoints=np.round(np.arange(.60,.9601,.02),10); centers=sorted(set(round(float(mid)+d,10) for mid in midpoints for d in (-.03,.03)))
src_dys={}
for c in centers:
    mask=C.smooth_inner_weight(np.asarray(mesh['Rq']),c,.02); sb=C.assemble_source(mesh,yb,mask,True); ss=C.assemble_source(mesh,ys,mask,False); src_dys[c]=(spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss))
# unmasked/full source state derivative for scale floor
ones=np.ones_like(np.asarray(mesh['Rq']),float); sb=C.assemble_source(mesh,yb,ones,True); ss=C.assemble_source(mesh,ys,ones,False); full_dy=(spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss))
xs=xs_for(xmax); cols=[]; maxstep=0.
for db,ds in beta_dys:
    g,sa=beta_profile(mesh,yb,ys,db,ds,xs); cols.append(g); maxstep=max(maxstep,sa)
U,sv,_=np.linalg.svd(np.column_stack(cols),full_matrices=False); orth=lambda g:g-U@(U.T@g)
gfull,sa=prof(mesh,yb,ys,full_dy[0],full_dy[1],xs,None,'full'); maxstep=max(maxstep,sa); full_orth=float(np.linalg.norm(orth(gfull))); floor=.10*full_orth
# full-only cumulative scan
data={}
for j,c in enumerate(centers):
    db,ds=src_dys[c]; g,sa=prof(mesh,yb,ys,db,ds,xs,c,'full'); data[c]=g; maxstep=max(maxstep,sa)
    if j%6==0: print('SCAN_CENTER',c,flush=True)
cands=[]
for mid in midpoints:
    a=round(float(mid)-.03,10); e=round(float(mid)+.03,10); inner=orth(data[a]); band=orth(data[e]-data[a]); ni=float(np.linalg.norm(inner)); nb=float(np.linalg.norm(band)); after=float(np.linalg.norm(inner+band)); angle=ang(inner,band); reduction=1-after/max(ni,1e-30); eligible=bool(ni>=floor and nb>=floor and angle>150 and reduction>=.5)
    cands.append({'midpoint':float(mid),'pair':[a,e],'distance_to_xmax':abs(float(mid)-xmax),'inner_orth_norm':ni,'band_orth_norm':nb,'angle_deg':angle,'reduction_fraction':reduction,'eligible':eligible})
elig=[c for c in cands if c['eligible']]; winner=max(elig,key=lambda c:c['reduction_fraction']) if elig else None; localization=bool(winner and winner['distance_to_xmax']<=.05)
mechanics=None; mechanics_pass=False; closure_scaled=float('nan')
if winner:
    a,e=winner['pair']; dd={}
    for center in (a,e):
        db,ds=src_dys[center]; dd[center]={}
        for kind in ('direct','state'):
            g,sa=prof(mesh,yb,ys,db,ds,xs,center,kind); dd[center][kind]=g; maxstep=max(maxstep,sa)
    bd=orth(dd[e]['direct']-dd[a]['direct']); bs=orth(dd[e]['state']-dd[a]['state']); bf=orth(data[e]-data[a]); nd=float(np.linalg.norm(bd)); ns=float(np.linalg.norm(bs)); nf=float(np.linalg.norm(bf)); ratio=nd/max(ns,1e-30); dsang=ang(bd,bs); residual_ratio=nf/max(nd,ns,1e-30); closure_scaled=float(np.linalg.norm(bf-bd-bs)/max(nd,ns,1e-30)); mechanics_pass=bool(dsang>160 and .75<=ratio<=1.33 and residual_ratio<=.35 and closure_scaled<1e-5)
    mechanics={'direct_state_angle_deg':dsang,'direct_over_state':ratio,'full_over_max_component':residual_ratio,'closure_over_max_component':closure_scaled,'pass':mechanics_pass}
valid=bool(maxstep<.01 and np.linalg.norm(gb,np.inf)<1e-7 and np.linalg.norm(gs,np.inf)<1e-7 and mechanics_pass)
case_pass=bool(valid and localization)
out={'xmax':xmax,'interface_singular_values':sv.tolist(),'full_orth_norm':full_orth,'scan_floor':floor,'winner':winner,'localization_pass':localization,'winner_mechanics':mechanics,'max_step_angle_deg':maxstep,'baseline_grad_inf_pre':float(np.linalg.norm(gb,np.inf)),'baseline_grad_inf_twist':float(np.linalg.norm(gs,np.inf)),'valid':valid,'pass':case_pass,'candidates':cands}
path=OUT/f'R190_CASE_XMAX_{xmax:.2f}.json'; path.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'path':str(path),'winner':winner,'localization_pass':localization,'mechanics':mechanics,'maxstep':maxstep,'valid':valid,'pass':case_pass},indent=2))
