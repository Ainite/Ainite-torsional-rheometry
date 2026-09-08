from pathlib import Path
import json, math, numpy as np
import scipy.sparse.linalg as spla
import r183_core as C
HERE=Path(__file__).resolve().parent; INP=HERE/'input'; OUT=HERE/'results'
mesh=C.m.build(C.NR,C.NZ,5.,.1,C.EDGE_CELLS,None,'tapered_springs'); z=np.load(INP/'baseline_edgewidth_0.10_aspect5_34x20.npz'); yb=np.asarray(z['yb']); ys=np.asarray(z['ys'])
_,_,Hb=C.m.assemble(mesh,yb,C.K,0.,C.EPS,True,*C.BETA); _,_,Hs=C.m.assemble(mesh,ys,C.K,C.GAM,C.EPS,True,*C.BETA)
def xs_for(xmax):
 j=np.arange(25.); return np.sqrt(.05**2+(j+.5)/25*(xmax*xmax-.05**2))
def samp(x,v,xs):
 o=np.argsort(x); return np.interp(xs,x[o],v[o])
def deriv(dyb,dys,xs,center,kind):
 vals={}
 for h in [1e-4,5e-5]:
  if kind=='full': ap=(yb+h*dyb,ys+h*dys,+h); am=(yb-h*dyb,ys-h*dys,-h)
  elif kind=='direct': ap=(yb,ys,+h); am=(yb,ys,-h)
  else: ap=(yb+h*dyb,ys+h*dys,0.); am=(yb-h*dyb,ys-h*dys,0.)
  xp,pp,_=C.raw_prof(mesh,ap[0],ap[1],ap[2],center,.02); xm,pm,_=C.raw_prof(mesh,am[0],am[1],am[2],center,.02)
  vals[h]=(samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h)
 return vals[5e-5]
def angle(a,b): return C.angle(a,b)
out=[]
for xmax in [.75,.85,.95]:
 xs=xs_for(xmax)
 # interface span
 cols=[]
 for ch in range(3):
  sb=C.beta_residual_derivative(mesh,yb,0.,ch); ss=C.beta_residual_derivative(mesh,ys,C.GAM,ch); db=spla.spsolve(Hb,-sb); ds=spla.spsolve(Hs,-ss)
  # state only beta
  vals={}
  for h in [5e-5]:
   xp,pp,_=C.raw_prof(mesh,yb+h*db,ys+h*ds,0.,None,.02); xm,pm,_=C.raw_prof(mesh,yb-h*db,ys-h*ds,0.,None,.02); vals[h]=(samp(xp,pp,xs)-samp(xm,pm,xs))/(2*h)
  cols.append(vals[5e-5])
 U,_,_=np.linalg.svd(np.column_stack(cols),full_matrices=False); orth=lambda g:g-U@(U.T@g)
 pp={}
 for center in [xmax-.03,xmax+.03]:
  mask=C.smooth_inner_weight(np.asarray(mesh['Rq']),center,.02); sb=C.assemble_source(mesh,yb,mask,True); ss=C.assemble_source(mesh,ys,mask,False); db=spla.spsolve(Hb,-sb); ds=spla.spsolve(Hs,-ss)
  pp[center]={k:deriv(db,ds,xs,center,k) for k in ['full','direct','state']}
 a=xmax-.03;e=xmax+.03
 inner=orth(pp[a]['full']); bands={k:orth(pp[e][k]-pp[a][k]) for k in ['full','direct','state']}
 rec={'xmax':xmax,'inner_orth_norm':float(np.linalg.norm(inner))}
 for k,v in bands.items(): rec[k]={'norm':float(np.linalg.norm(v)),'angle_to_inner':angle(inner,v)}
 rec['direct_state_band_angle']=angle(bands['direct'],bands['state']); rec['closure_rel']=float(np.linalg.norm(bands['full']-bands['direct']-bands['state'])/np.linalg.norm(bands['full']))
 out.append(rec)
print(json.dumps(out,indent=2)); (OUT/'R183_POSTHOC_DIRECT_STATE.json').write_text(json.dumps({'status':'post-hoc explanatory decomposition','cases':out},indent=2)+'\n')
