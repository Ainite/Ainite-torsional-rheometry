#!/usr/bin/env python3
import json, importlib.util, sys, math
from pathlib import Path
import numpy as np, scipy.sparse.linalg as spla
import r195_core as C
HERE=Path(__file__).resolve().parent; INP=HERE/'input'; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)

def xs_for(xmax):
    j=np.arange(25.); return np.sqrt(.05**2+(j+.5)/25*(xmax*xmax-.05**2))
def samp(x,v,xs):
    o=np.argsort(x); return np.interp(xs,np.asarray(x)[o],np.asarray(v)[o])
def prof(mesh,yb,ys,dyb,dys,xs,center,halfwidth):
    vals={}
    for h in [1e-4,5e-5]:
        xp,pp,_=C.raw_prof(mesh,yb+h*dyb,ys+h*dys,+h,center,halfwidth); xm,pm,_=C.raw_prof(mesh,yb-h*dyb,ys-h*dys,-h,center,halfwidth)
        vals[h]=(samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h)
    return vals[5e-5],C.angle(vals[1e-4],vals[5e-5])
def beta_prof(mesh,yb,ys,db,ds,xs):
    vals={}
    for h in [1e-4,5e-5]:
        xp,pp,_=C.raw_prof(mesh,yb+h*db,ys+h*ds,0.,None,.02); xm,pm,_=C.raw_prof(mesh,yb-h*db,ys-h*ds,0.,None,.02)
        vals[h]=(samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h)
    return vals[5e-5],C.angle(vals[1e-4],vals[5e-5])
mesh=C.m.build(C.NR,C.NZ,5.,.1,C.EDGE_CELLS,None,'tapered_springs'); z=np.load(INP/'baseline_edgewidth_0.10_aspect5_34x20.npz'); yb=np.asarray(z['yb']); ys=np.asarray(z['ys'])
_,gb,Hb=C.m.assemble(mesh,yb,C.K,0.,C.EPS,True,*C.BETA); _,gs,Hs=C.m.assemble(mesh,ys,C.K,C.GAM,C.EPS,True,*C.BETA)
beta_dys=[]
for ch in range(3):
    sb=C.beta_residual_derivative(mesh,yb,0.,ch); ss=C.beta_residual_derivative(mesh,ys,C.GAM,ch); beta_dys.append((spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss)))
# local cell geometry
rp=np.asarray(mesh['rp'],float)/float(C.m.R); widths=np.diff(rp)
def cell_info(x):
    i=int(np.searchsorted(rp,x,side='right')-1); i=max(0,min(i,len(widths)-1)); return {'cell_index':i,'lower':float(rp[i]),'upper':float(rp[i+1]),'width':float(widths[i])}
res={'schema':'U1-R195-resolution-projection-control-v1','radial_knots':rp.tolist(),'cases':[],'outer_extension':None,'max_step_angle_deg':0.0,'equilibrium_grad_inf_pre':float(np.linalg.norm(gb,np.inf)),'equilibrium_grad_inf_twist':float(np.linalg.norm(gs,np.inf))}
for xmax,mid in [(0.73,0.74),(0.83,0.86),(0.93,0.96)]:
    xs=xs_for(xmax); cols=[]
    for db,ds in beta_dys:
        g,sa=beta_prof(mesh,yb,ys,db,ds,xs); cols.append(g); res['max_step_angle_deg']=max(res['max_step_angle_deg'],sa)
    U,sv,_=np.linalg.svd(np.column_stack(cols),full_matrices=False)
    case={'xmax':xmax,'historical_winner_midpoint':mid,'distance_to_xmax':abs(mid-xmax),'cell':cell_info(mid),'distance_in_local_cell_widths':abs(mid-xmax)/cell_info(mid)['width'],'interface_singular_values':sv.tolist(),'width_sensitivity':[]}
    for d in [0.01,0.02,0.04]:
        ctrs=[mid-.03,mid+.03]; dat={}
        for c in ctrs:
            mask=C.smooth_inner_weight(np.asarray(mesh['Rq']),c,d); sb=C.assemble_source(mesh,yb,mask,True); ss=C.assemble_source(mesh,ys,mask,False); db=spla.spsolve(Hb,-sb); ds=spla.spsolve(Hs,-ss); g,sa=prof(mesh,yb,ys,db,ds,xs,c,d); dat[c]=g; res['max_step_angle_deg']=max(res['max_step_angle_deg'],sa)
        fullrank={}
        for rank in [2,3]:
            Ur=U[:,:rank]; orth=lambda g:g-Ur@(Ur.T@g); inner=orth(dat[ctrs[0]]); band=orth(dat[ctrs[1]]-dat[ctrs[0]]); after=inner+band; ni=np.linalg.norm(inner); nb=np.linalg.norm(band)
            fullrank[str(rank)]={'inner_orth_norm':float(ni),'band_orth_norm':float(nb),'after_orth_norm':float(np.linalg.norm(after)),'angle_deg':C.angle(inner,band),'reduction_fraction':float(1-np.linalg.norm(after)/max(ni,1e-30))}
        case['width_sensitivity'].append({'halfwidth':d,'rank_results':fullrank})
    res['cases'].append(case)
# Outer extension at xmax=.93, d=.01.  Center 1.01 has lo=1.00, so at all interior quadrature points its cumulative weight is effectively full-radius.
xmax=.93; xs=xs_for(xmax); cols=[]
for db,ds in beta_dys:
    g,_=beta_prof(mesh,yb,ys,db,ds,xs); cols.append(g)
U,sv,_=np.linalg.svd(np.column_stack(cols),full_matrices=False); orth=lambda g:g-U@(U.T@g)
centers=sorted(set([round(mid-.03,10) for mid in [0.92,0.94,0.96,0.98]]+[round(mid+.03,10) for mid in [0.92,0.94,0.96,0.98]]))
dat={}
for c in centers:
    mask=C.smooth_inner_weight(np.asarray(mesh['Rq']),c,.01); sb=C.assemble_source(mesh,yb,mask,True); ss=C.assemble_source(mesh,ys,mask,False); db=spla.spsolve(Hb,-sb); ds=spla.spsolve(Hs,-ss); g,sa=prof(mesh,yb,ys,db,ds,xs,c,.01); dat[c]=g; res['max_step_angle_deg']=max(res['max_step_angle_deg'],sa)
out=[]
for mid in [0.92,0.94,0.96,0.98]:
    lo=round(mid-.03,10); hi=round(mid+.03,10); inner=orth(dat[lo]); band=orth(dat[hi]-dat[lo]); ni=np.linalg.norm(inner); after=np.linalg.norm(inner+band); out.append({'midpoint':mid,'pair':[lo,hi],'angle_deg':C.angle(inner,band),'reduction_fraction':float(1-after/max(ni,1e-30)),'inner_orth_norm':float(ni),'after_orth_norm':float(after)})
res['outer_extension']={'xmax':.93,'halfwidth':.01,'candidates':out,'best':max(out,key=lambda x:x['reduction_fraction'])}
(OUT/'R195_RESOLUTION_PROJECTION_CONTROL.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps({'cases':res['cases'],'outer_extension':res['outer_extension'],'max_step_angle_deg':res['max_step_angle_deg']},indent=2))
