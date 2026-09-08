#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import importlib.util, sys, json
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; INP=ROOT/'input'; OUT=ROOT/'meshcheck_30x40'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(INP)); import postprocess_core as post
BASE_BETA=np.array([1000.,3000.,3000.]); C_PROBE=0.7644117178
OLD="""    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    Wi=.5*(Js**(-2/3)*I1-3.)\n"""
MR="""    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    C2=C@C; I2=.5*(I1*I1-jnp.trace(C2,axis1=-2,axis2=-1))\n    f=0.30*0.95/(1.0-0.30*(1.0-0.95))\n    I1b=Js**(-2/3)*I1; I2b=Js**(-4/3)*I2\n    Wi=.5*(1-f)*(I1b-3.)+.5*f*(I2b-3.)\n"""
PROBE="""    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    C2=C@C; I2=.5*(I1*I1-jnp.trace(C2,axis1=-2,axis2=-1))\n    f=0.30*0.95/(1.0-0.30*(1.0-0.95))\n    I1b=Js**(-2/3)*I1; I2b=Js**(-4/3)*I2\n    lam=0.95\n    I10=lam*lam+2.0/lam; I20=2.0*lam+lam**(-2)\n    h=(I1b-I10)-lam*(I2b-I20)\n    cprobe=0.7644117178\n    Wi=.5*(1-f)*(I1b-3.)+.5*f*(I2b-3.)+cprobe*h*h\n"""
def load(name,block):
    src=(INP/'solver_core.py').read_text(); tmp=OUT/f'_{name}.py'; tmp.write_text(src.replace(OLD,block))
    sp=importlib.util.spec_from_file_location(name,tmp); m=importlib.util.module_from_spec(sp); assert sp.loader; sp.loader.exec_module(m); return m
def build(m): return m.build(30,40,5.0,0.1,8,None,'tapered_springs')
def continuation(m,mesh,beta=BASE_BETA):
    y=m.initial(mesh)
    for e in np.arange(.005,.0501,.005):
        y,ok,h=m.newton(mesh,y,10000.,0.,float(e),*beta,tol=1e-8,maxit=30); print('pre',float(e),ok,h[-1][0],flush=True)
        if not ok: raise RuntimeError(('pre',e,h[-1]))
    yb=y.copy()
    for g in [.05,.10,.15,.20]:
        y,ok,h=m.newton(mesh,y,10000.,g,.05,*beta,tol=1e-8,maxit=30); print('tw',g,ok,h[-1][0],flush=True)
        if not ok: raise RuntimeError(('tw',g,h[-1]))
    return yb,y.copy()
def solve(m,mesh,yb0,ys0,beta):
    yb,okb,hb=m.newton(mesh,yb0.copy(),10000.,0.,.05,*beta,tol=1e-8,maxit=30)
    ys,oks,hs=m.newton(mesh,ys0.copy(),10000.,.2,.05,*beta,tol=1e-8,maxit=30)
    print('state',beta.tolist(),okb,oks,hb[-1][0],hs[-1][0],flush=True)
    if not(okb and oks): raise RuntimeError(('state',beta.tolist(),hb[-1],hs[-1]))
    return yb,ys
def raw(m,mesh,yb,ys,c):
    p0=post.top_stress(m,mesh,yb,0.,.05,post.F_MR,c,0.,0.); pg=post.top_stress(m,mesh,ys,.2,.05,post.F_MR,c,0.,0.)
    coords,_=m.unpack(mesh,ys,.2,.05); top=np.isclose(mesh['X'][:,1],mesh['H']); a=float(np.max(coords[top,0])); o=np.argsort(p0['r_cur'])
    base=np.interp(pg['r_cur'],p0['r_cur'][o],p0['szz_Pa'][o]); return np.asarray(pg['r_cur'])/a, -(np.asarray(pg['szz_Pa'])-base), a
def pts():
    j=np.arange(25.); return np.sqrt(.05**2+(j+.5)/25*(.85**2-.05**2))
def samp(x,y,xs):
    o=np.argsort(x); return np.interp(xs,np.asarray(x)[o],np.asarray(y)[o])
def met(c,f,ac,af):
    eta=float(c@f/(f@f)); cos=float(c@f/(np.linalg.norm(c)*np.linalg.norm(f)))
    return {'constitutive_norm_sigma':float(np.linalg.norm(c)/2),'interface_norm_sigma':float(np.linalg.norm(f)/2),'direct_residual_sigma':float(np.linalg.norm(c-f)/2),'direction_angle_deg':float(np.degrees(np.arccos(np.clip(cos,-1,1)))),'optimal_scalar':eta,'direction_profiled_residual_sigma':float(np.linalg.norm(c-eta*f)/2),'radius_mismatch_um':float((af-ac)*30000.)}
mr=load('m30mr',MR); probe=load('m30probe',PROBE); mm=build(mr); mp=build(probe)
print('DOF',len(mr.initial(mm)),'edge_cells',mm['edge_cells'],flush=True)
yb0,ys0=continuation(mr,mm); np.savez_compressed(OUT/'baseline_state_30x40_ec8.npz',yb0=yb0,ys0=ys0)
ybc,ysc=solve(probe,mp,yb0,ys0,BASE_BETA)
xb,pb,abase=raw(mr,mm,yb0,ys0,0.); xc,pc,ac=raw(probe,mp,ybc,ysc,C_PROBE); xs=pts(); base=samp(xb,pb,xs); con=samp(xc,pc,xs)-base
rec={'mesh':'30x40_ec8','ndof':len(yb0),'baseline_radius_nondim':abase,'constitutive_radius_nondim':ac,'constitutive_norm_sigma':float(np.linalg.norm(con)/2),'constitutive_J_pre':probe.jacobian_range(mp,ybc,0.,.05),'constitutive_J_twist':probe.jacobian_range(mp,ysc,.2,.05)}
np.savez_compressed(OUT/'finite_constitutive_state_30x40.npz',yb=ybc,ys=ysc,c=C_PROBE)
profiles={'x':xs,'finite_constitutive_shift_Pa':con}
for b,tag in [(.25,'025'),(.32,'032')]:
    beta=BASE_BETA*np.exp(np.array([-b,b,b])); yb,ys=solve(mr,mm,yb0,ys0,beta); xf,pf,af=raw(mr,mm,yb,ys,0.); f=samp(xf,pf,xs)-base
    rec[tag]=met(con,f,ac,af); rec[tag].update({'betas':beta.tolist(),'J_pre':mr.jacobian_range(mm,yb,0.,.05),'J_twist':mr.jacobian_range(mm,ys,.2,.05)})
    profiles[f'interface_{tag}_shift_Pa']=f; np.savez_compressed(OUT/f'coordinated_interface_{tag}_state_30x40.npz',yb=yb,ys=ys,betas=beta)
X=np.c_[np.ones(25),xs*xs]; rec['finite_constitutive_dAB_Pa']=np.linalg.lstsq(X,con,rcond=None)[0].tolist()
for tag in ['025','032']: rec[f'interface_{tag}_dAB_Pa']=np.linalg.lstsq(X,profiles[f'interface_{tag}_shift_Pa'],rcond=None)[0].tolist()
(OUT/'meshcheck_30x40.json').write_text(json.dumps(rec,indent=2)+'\n')
np.savetxt(OUT/'profiles_30x40.csv',np.column_stack(list(profiles.values())),delimiter=',',header=','.join(profiles),comments='')
print(json.dumps(rec,indent=2),flush=True)
