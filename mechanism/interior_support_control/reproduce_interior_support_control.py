from pathlib import Path
import sys,json, numpy as np, scipy.sparse.linalg as spla
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'absolute_radius_self_localization'
RESULTS=HERE/'results'; RESULTS.mkdir(exist_ok=True); sys.path.insert(0,str(ROOT)); import r190_core as C
mesh=C.m.build(C.NR,C.NZ,5.,.1,C.EDGE_CELLS,None,'tapered_springs'); z=np.load(ROOT/'input'/'baseline_edgewidth_0.10_aspect5_34x20.npz'); yb=np.asarray(z['yb']); ys=np.asarray(z['ys'])
_,_,Hb=C.m.assemble(mesh,yb,C.K,0.,C.EPS,True,*C.BETA); _,_,Hs=C.m.assemble(mesh,ys,C.K,C.GAM,C.EPS,True,*C.BETA)
def xs_for(xmax):
 j=np.arange(25.); return np.sqrt(.05**2+(j+.5)/25*(xmax*xmax-.05**2))
def samp(x,v,xs):
 o=np.argsort(x); return np.interp(xs,np.asarray(x)[o],np.asarray(v)[o])
def tangent(dyb,dys,center,xs):
 vals=[]
 for h in [1e-4,5e-5]:
  xp,pp,_=C.raw_prof(mesh,yb+h*dyb,ys+h*dys,+h,center,.02); xm,pm,_=C.raw_prof(mesh,yb-h*dyb,ys-h*dys,-h,center,.02)
  vals.append((samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h))
 return vals[-1],C.angle(vals[0],vals[-1])
def beta_prof(db,ds,xs):
 vals=[]
 for h in [1e-4,5e-5]:
  xp,pp,_=C.raw_prof(mesh,yb+h*db,ys+h*ds,0.,None,.02); xm,pm,_=C.raw_prof(mesh,yb-h*db,ys-h*ds,0.,None,.02)
  vals.append((samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h))
 return vals[-1]
# full and inner support source derivatives
ones=np.ones_like(np.asarray(mesh['Rq']),float)
inner=C.smooth_inner_weight(np.asarray(mesh['Rq']),.60,.02)
dysrc={}
for name,mask in [('full',ones),('inner60',inner)]:
 sb=C.assemble_source(mesh,yb,mask,True); ss=C.assemble_source(mesh,ys,mask,False)
 dysrc[name]=(spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss))
# beta derivatives once
bds=[]
for ch in range(3):
 sb=C.beta_residual_derivative(mesh,yb,0.,ch); ss=C.beta_residual_derivative(mesh,ys,C.GAM,ch); bds.append((spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss)))
out={'rule':{'support':'C2 inner cutoff center 0.60 halfwidth 0.02','support_effect_pass':'inner orth fraction >= max(2*full orth fraction,0.08)','observers':[.73,.83,.93]},'cases':{}}
for xmax in [.73,.83,.93]:
 xs=xs_for(xmax); B=np.column_stack([beta_prof(*bd,xs) for bd in bds]); U,sv,_=np.linalg.svd(B,full_matrices=False); U2=U[:,:2]; U3=U[:,:3]
 case={'interface_singular_values':sv.tolist()}
 for name in ['full','inner60']:
  g,step=tangent(*dysrc[name],None if name=='full' else .60,xs)
  n=np.linalg.norm(g); r2=g-U2@(U2.T@g); r3=g-U3@(U3.T@g)
  case[name]={'norm':float(n),'rank2_orth_fraction':float(np.linalg.norm(r2)/n),'rank3_orth_fraction':float(np.linalg.norm(r3)/n),'rank3_orth_norm':float(np.linalg.norm(r3)),'step_angle_deg':float(step)}
 f=case['full']['rank3_orth_fraction']; q=case['inner60']['rank3_orth_fraction']; case['support_effect_pass']=bool(q>=max(2*f,.08)); case['ratio_inner_to_full']=float(q/f)
 out['cases'][f'{xmax:.2f}']=case
 print(xmax,'full',f,'inner',q,'ratio',q/f,'pass',case['support_effect_pass'])
out['verdict']='INNER_SUPPORT_EFFECT_SUPPORTED' if all(c['support_effect_pass'] for c in out['cases'].values()) else 'INNER_SUPPORT_EFFECT_NOT_UNIFORMLY_SUPPORTED'
(RESULTS/'interior_support_control.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
