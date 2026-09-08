#!/usr/bin/env python3
import argparse,json,numpy as np
from pathlib import Path
import scipy.sparse.linalg as spla
import r194_core as C
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
def beta_prof(mesh,yb,ys,db,ds,xs):
    vals={}
    for h in [1e-4,5e-5]:
        # beta tangent has no direct constitutive term: perturb states only.
        xp,pp,_=C.raw_prof(mesh,yb+h*db,ys+h*ds,0.,None,.02); xm,pm,_=C.raw_prof(mesh,yb-h*db,ys-h*ds,0.,None,.02)
        vals[h]=(samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h)
    return vals[5e-5],C.angle(vals[1e-4],vals[5e-5])
ap=argparse.ArgumentParser(); ap.add_argument('--xmax',type=float,required=True); a=ap.parse_args(); xmax=a.xmax
mesh=C.m.build(C.NR,C.NZ,5.,.1,C.EDGE_CELLS,None,'tapered_springs'); z=np.load(INP/'baseline_edgewidth_0.10_aspect5_34x20.npz'); yb=np.asarray(z['yb']); ys=np.asarray(z['ys'])
_,gb,Hb=C.m.assemble(mesh,yb,C.K,0.,C.EPS,True,*C.BETA); _,gs,Hs=C.m.assemble(mesh,ys,C.K,C.GAM,C.EPS,True,*C.BETA)
# Interface nuisance basis recomputed for this observer.
beta_dys=[]
for ch in range(3):
    sb=C.beta_residual_derivative(mesh,yb,0.,ch); ss=C.beta_residual_derivative(mesh,ys,C.GAM,ch); beta_dys.append((spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss)))
midpoints=np.round(np.arange(.60,.9601,.02),10); centers=sorted(set(round(float(mid)+d,10) for mid in midpoints for d in (-.03,.03)))
src_dys={}
for c in centers:
    mask=C.smooth_inner_weight(np.asarray(mesh['Rq']),c,.02); sb=C.assemble_partition_source(mesh,yb,mask,True); ss=C.assemble_partition_source(mesh,ys,mask,False); src_dys[c]=(spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss))
ones=np.ones_like(np.asarray(mesh['Rq']),float); sb=C.assemble_partition_source(mesh,yb,ones,True); ss=C.assemble_partition_source(mesh,ys,ones,False); full_dy=(spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss))
xs=xs_for(xmax); cols=[]; maxstep=0.
for db,ds in beta_dys:
    g,sa=beta_prof(mesh,yb,ys,db,ds,xs); cols.append(g); maxstep=max(maxstep,sa)
U,sv,_=np.linalg.svd(np.column_stack(cols),full_matrices=False); orth=lambda g:g-U@(U.T@g)
gfull,sa=prof(mesh,yb,ys,full_dy[0],full_dy[1],xs,None,'full'); maxstep=max(maxstep,sa); fullorth=float(np.linalg.norm(orth(gfull))); floor=.1*fullorth
data={}
for c in centers:
    db,ds=src_dys[c]; g,sa=prof(mesh,yb,ys,db,ds,xs,c,'full'); data[c]=g; maxstep=max(maxstep,sa)
cands=[]
for mid in midpoints:
    lo=round(float(mid)-.03,10); hi=round(float(mid)+.03,10); interior=orth(data[lo]); band=orth(data[hi]-data[lo]); ni=float(np.linalg.norm(interior)); nb=float(np.linalg.norm(band)); after=float(np.linalg.norm(interior+band)); ang=C.angle(interior,band); red=1-after/max(ni,1e-30); eligible=bool(ni>=floor and nb>=floor and ang>150 and red>=.5)
    cands.append({'midpoint':float(mid),'pair':[lo,hi],'distance_to_xmax':abs(float(mid)-xmax),'inner_orth_norm':ni,'band_orth_norm':nb,'after_orth_norm':after,'angle_deg':ang,'reduction_fraction':red,'eligible':eligible})
elig=[x for x in cands if x['eligible']]; winner=max(elig,key=lambda x:x['reduction_fraction']) if elig else None; loc=bool(winner and winner['distance_to_xmax']<=.05)
out={'schema':'U1-R194-MR-partition-case-v1','source':'Mooney-Rivlin coefficient-partition dW/df','xmax':xmax,'interface_singular_values':sv.tolist(),'full_orth_norm':fullorth,'scan_floor':floor,'winner':winner,'localization_pass':loc,'max_step_angle_deg':maxstep,'equilibrium_grad_inf_pre':float(np.linalg.norm(gb,np.inf)),'equilibrium_grad_inf_twist':float(np.linalg.norm(gs,np.inf)),'numerically_valid':bool(maxstep<.01 and np.linalg.norm(gb,np.inf)<1e-7 and np.linalg.norm(gs,np.inf)<1e-7),'candidates':cands}
(OUT/f'R194_MR_CASE_XMAX_{xmax:.2f}.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'xmax':xmax,'full_orth_norm':fullorth,'winner':winner,'localization':loc,'maxstep':maxstep},indent=2))
