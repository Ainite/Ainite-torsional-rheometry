from pathlib import Path
import sys,json,math
import numpy as np
import scipy.sparse.linalg as spla
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'absolute_radius_self_localization'
RESULTS=HERE/'results'; RESULTS.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
import r190_core as C

def xs_for(xmax):
    j=np.arange(25.); return np.sqrt(.05**2+(j+.5)/25*(xmax*xmax-.05**2))
def samp(x,v,xs):
    o=np.argsort(x); return np.interp(xs,np.asarray(x)[o],np.asarray(v)[o])
def raw_tangent(mesh,yb,ys,dyb,dys,center):
    vals=[]
    for h in [1e-4,5e-5]:
        xp,pp,_=C.raw_prof(mesh,yb+h*dyb,ys+h*dys,+h,center,.02)
        xm,pm,_=C.raw_prof(mesh,yb-h*dyb,ys-h*dys,-h,center,.02)
        # common normalized radius grid = union approx; use plus x, interp minus
        pm_i=np.interp(xp,np.asarray(xm)[np.argsort(xm)],np.asarray(pm)[np.argsort(xm)])
        vals.append((xp,(pp-pm_i)/(2*h)))
    return vals[-1]

mesh=C.m.build(C.NR,C.NZ,5.,.1,C.EDGE_CELLS,None,'tapered_springs')
z=np.load(ROOT/'input'/'baseline_edgewidth_0.10_aspect5_34x20.npz'); yb=np.asarray(z['yb']); ys=np.asarray(z['ys'])
_,gb,Hb=C.m.assemble(mesh,yb,C.K,0.,C.EPS,True,*C.BETA); _,gs,Hs=C.m.assemble(mesh,ys,C.K,C.GAM,C.EPS,True,*C.BETA)
midpoints=np.round(np.arange(.60,.9601,.02),10)
centers=sorted(set(round(float(mid)+d,10) for mid in midpoints for d in (-.03,.03)))
print('assembling source states',len(centers),flush=True)
source={}
for j,c in enumerate(centers):
    mask=C.smooth_inner_weight(np.asarray(mesh['Rq']),c,.02)
    sb=C.assemble_source(mesh,yb,mask,True); ss=C.assemble_source(mesh,ys,mask,False)
    dyb=spla.spsolve(Hb,-sb); dys=spla.spsolve(Hs,-ss)
    source[c]=(dyb,dys)
    if j%8==0: print(' source',j,c,flush=True)
# full source raw tangent
ones=np.ones_like(np.asarray(mesh['Rq']),float); sb=C.assemble_source(mesh,yb,ones,True); ss=C.assemble_source(mesh,ys,ones,False)
full=(spla.spsolve(Hb,-sb),spla.spsolve(Hs,-ss))
print('postprocessing cumulative tangents',flush=True)
raw={}
for j,c in enumerate(centers):
    raw[c]=raw_tangent(mesh,yb,ys,*source[c],c)
    if j%8==0: print(' raw',j,c,flush=True)
raw_full=raw_tangent(mesh,yb,ys,*full,None)
# input mask matrix for right-space source-weight diagnostic
Rq=np.asarray(mesh['Rq']).ravel(); W=np.asarray(mesh['W']).ravel()
# each band mask is cumulative(e)-cumulative(a)
A=[]
for mid in midpoints:
    a=round(float(mid)-.03,10); e=round(float(mid)+.03,10)
    ma=C.smooth_inner_weight(np.asarray(mesh['Rq']),a,.02).ravel(); me=C.smooth_inner_weight(np.asarray(mesh['Rq']),e,.02).ravel()
    A.append(me-ma)
A=np.column_stack(A)
# target = uniform source on scan-covered annular region via outer-inner cumulative difference
mtarget=C.smooth_inner_weight(np.asarray(mesh['Rq']),.99,.02).ravel()-C.smooth_inner_weight(np.asarray(mesh['Rq']),.57,.02).ravel()
sw=np.sqrt(np.maximum(W,0))
coef=np.linalg.lstsq(A*sw[:,None],mtarget*sw,rcond=None)[0]
mask_rel=np.linalg.norm((A@coef-mtarget)*sw)/np.linalg.norm(mtarget*sw)
out={'input_mask_fit_relerr':float(mask_rel),'input_band_coefficients':coef.tolist(),'cases':{}}
for xmax in [0.73,0.83,0.93]:
    xs=xs_for(xmax); cols=[]
    for mid in midpoints:
        a=round(float(mid)-.03,10); e=round(float(mid)+.03,10)
        xa,ga=raw[a]; xe,ge=raw[e]
        va=samp(xa,ga,xs); ve=samp(xe,ge,xs)
        cols.append(ve-va)
    B=np.column_stack(cols)
    U,s,Vt=np.linalg.svd(B,full_matrices=False)
    energy=s*s; efrac=np.cumsum(energy)/np.sum(energy)
    # effective rank / participation ratio on singular-value energy
    p=energy/energy.sum(); erank=float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))); pr=float(1/np.sum(p*p))
    xf,gf=raw_full; gfull=samp(xf,gf,xs)
    proj2=U[:,:2]@(U[:,:2].T@gfull)
    resp2=float(np.linalg.norm(proj2)/np.linalg.norm(gfull))
    # actual scan-region source coefficient expansion in input singular basis
    q=Vt@coef
    qenergy=q*q; qfrac=np.cumsum(qenergy)/np.sum(qenergy)
    out['cases'][f'{xmax:.2f}']={
      'singular_values':s.tolist(),
      's2_over_s1':float(s[1]/s[0]),
      's3_over_s1':float(s[2]/s[0]),
      'first2_operator_energy_fraction':float(efrac[1]),
      'first3_operator_energy_fraction':float(efrac[2]),
      'entropy_effective_rank':erank,
      'participation_rank':pr,
      'full_Q1_response_fraction_in_first2_left_modes':resp2,
      'scan_region_Q1_weight_fraction_in_first2_right_modes':float(qfrac[1]),
      'scan_region_Q1_weight_fraction_in_first3_right_modes':float(qfrac[2]),
      'matrix_fro_norm':float(np.linalg.norm(B)),
    }
    np.savez(RESULTS/f'band_matrix_xmax_{xmax:.2f}.npz',B=B,midpoints=midpoints,xs=xs,singular_values=s,U=U,Vt=Vt,source_coeff=coef)
    print('xmax',xmax,'s ratios',s[1]/s[0],s[2]/s[0],'E2',efrac[1],'erank',erank,'resp2',resp2,'q2',qfrac[1],flush=True)
(RESULTS/'band_operator_svd.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
