from pathlib import Path
import json, math
import numpy as np
import scipy.sparse.linalg as spla
import r182_core as C
HERE=Path(__file__).resolve().parent; OUT=HERE/'results'

def proj_span_from_csv(path):
 d=np.genfromtxt(path,delimiter=',',names=True); B=np.c_[d['beta_r_log'],d['beta_theta_log'],d['beta_z_log']]; U,_,_=np.linalg.svd(B,full_matrices=False); return U

def run(w):
 mesh=C.m.build(C.NR,C.NZ,5.0,w,C.EDGE_CELLS,None,'tapered_springs'); z=np.load(OUT/f'baseline_edgewidth_{w:.2f}.npz'); yb=np.asarray(z['yb']); ys=np.asarray(z['ys']); U=proj_span_from_csv(OUT/f'profiles_edgewidth_{w:.2f}.csv'); orth=lambda g:g-U@(U.T@g)
 _,_,Hb=C.m.assemble(mesh,yb,C.K,0.,C.EPS,True,*C.BETA); _,_,Hs=C.m.assemble(mesh,ys,C.K,C.GAM,C.EPS,True,*C.BETA)
 prof={}
 for c in [.82,.88]:
  mask=C.smooth_inner_weight(np.asarray(mesh['Rq']),c,.02); sb=C.assemble_source(mesh,yb,mask,True); ss=C.assemble_source(mesh,ys,mask,False); dyb=spla.spsolve(Hb,-sb); dys=spla.spsolve(Hs,-ss); g,sa,_=C.tangent_profile(mesh,yb,ys,dyb,dys,1.,c,.02); prof[c]=g
 a=prof[.82]; e=prof[.88]; band=e-a; oa=orth(a); ob=orth(band); oe=orth(e)
 return {'edge_width':w,'transition_onset':1-w,'absolute_pair':[.82,.88],'orth_angle_deg':C.angle(oa,ob),'orth_norm_before':float(np.linalg.norm(oa)),'orth_norm_added_band':float(np.linalg.norm(ob)),'orth_norm_after':float(np.linalg.norm(oe)),'reduction_fraction':1-float(np.linalg.norm(oe)/np.linalg.norm(oa))}
rec={'schema':'U1-R182-posthoc-absolute-location-diagnostic-v1','status':'post-hoc diagnostic; does not alter preregistered R182 verdict','cases':[run(.05),run(.15)]}
(OUT/'R182_POSTHOC_ABSOLUTE_LOCATION.json').write_text(json.dumps(rec,indent=2)+'\n'); print(json.dumps(rec,indent=2))
