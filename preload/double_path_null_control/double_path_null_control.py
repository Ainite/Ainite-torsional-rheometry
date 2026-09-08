#!/usr/bin/env python3
from pathlib import Path
import importlib.util, sys, json, math
import numpy as np, scipy.sparse.linalg as spla
import jax, jax.numpy as jnp
HERE=Path(__file__).resolve().parent; INP=HERE/'input'; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(INP)); import postprocess_core as post
K=10000.; EPS=.05; GAM=.2; BETA=np.array([1000.,3000.,3000.]); SIGMA=2.0
OLD="""    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    Wi=.5*(Js**(-2/3)*I1-3.)\n"""
MR="""    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    C2=C@C; I2=.5*(I1*I1-jnp.trace(C2,axis1=-2,axis2=-1))\n    f=0.30*0.95/(1.0-0.30*(1.0-0.95))\n    I1b=Js**(-2/3)*I1; I2b=Js**(-4/3)*I2\n    Wi=.5*(1-f)*(I1b-3.)+.5*f*(I2b-3.)\n"""
src=(INP/'solver_core.py').read_text(); tmp=OUT/'_r196_mr.py'; tmp.write_text(src.replace(OLD,MR)); sp=importlib.util.spec_from_file_location('r196_mr',tmp); m=importlib.util.module_from_spec(sp); assert sp.loader; sp.loader.exec_module(m)
# Restore exact primary 38x40 graded radial mesh.
orig=m.radial_knots; R=m.R
def radial_knots(nr,edge_width=.1,edge_cells=None,grading_power=None):
    if int(nr)==38 and int(edge_cells or 0)==16:
        bulk=22; r0=R*(1-edge_width); bulkarr=np.linspace(0.,r0,bulk+1); u=np.arange(1,17,dtype=float)/16.; edge=r0+R*edge_width*(1-(1-u)**2); return np.r_[bulkarr,edge]
    return orig(nr,edge_width,edge_cells,grading_power)
m.radial_knots=radial_knots
mesh=m.build(38,40,5.,.1,16,None,'tapered_springs'); z=np.load(INP/'baseline_state_38x40.npz'); yb=np.asarray(z['yb0']); ys=np.asarray(z['ys0'])
_,gb,Hb=m.assemble(mesh,yb,K,0.,EPS,True,*BETA); _,gs,Hs=m.assemble(mesh,ys,K,GAM,EPS,True,*BETA)

def hfun(I1,I2,lam): return I1-lam*I2+lam*lam-1./lam

def _src_terms(z,N,DG,Rq,W):
    ve=z[:27].reshape(9,3); val=jnp.einsum('gi,ic->gc',N,ve); der=jnp.einsum('gia,ic->gca',DG,ve)
    rr=val[:,0]; rR=der[:,0,0]; rz=der[:,0,1]; pR=der[:,1,0]; pz=der[:,1,1]; zR=der[:,2,0]; zz=der[:,2,1]; zero=jnp.zeros_like(rr)
    F=jnp.stack([jnp.stack([rR,zero,rz],1),jnp.stack([rr*pR,rr/Rq,rr*pz],1),jnp.stack([zR,zero,zz],1)],1); J=jnp.linalg.det(F); Js=jnp.maximum(J,1e-10)
    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1); C2=C@C; I2=.5*(I1*I1-jnp.trace(C2,axis1=-2,axis2=-1)); I1b=Js**(-2/3)*I1; I2b=Js**(-4/3)*I2
    h95=I1b-.95*I2b+.95**2-1/.95; h85=I1b-.85*I2b+.85**2-1/.85
    return h95,h85,W
def src_energy_single(z,N,DG,Rq,W):
    h95,h85,W=_src_terms(z,N,DG,Rq,W); return jnp.sum(h95*h95*W)
def src_energy_double(z,N,DG,Rq,W):
    h95,h85,W=_src_terms(z,N,DG,Rq,W); return jnp.sum(h95*h95*h85*h85*W)
src_grad_single=jax.jit(jax.vmap(jax.grad(src_energy_single),in_axes=(0,0,0,0,0)))
src_grad_double=jax.jit(jax.vmap(jax.grad(src_energy_double),in_axes=(0,0,0,0,0)))
def assemble_source(y,is_pre,kind):
    lv=m.local_vars(mesh,y,0.,EPS) if is_pre else m.local_vars(mesh,y,GAM,EPS); gradfun=src_grad_single if kind==0 else src_grad_double; gl=np.asarray(gradfun(jnp.asarray(lv),jnp.asarray(mesh['N']),jnp.asarray(mesh['G']),jnp.asarray(mesh['Rq']),jnp.asarray(mesh['W']))); gg=np.zeros(len(y))
    for e,lm in enumerate(mesh['maps']):
        ok=lm>=0; np.add.at(gg,lm[ok],gl[e,ok])
    return gg

def perturb_P(F,kind,c):
    J=np.linalg.det(F); C=F.T@F; I1=np.trace(C); I2=.5*(I1*I1-np.trace(C@C)); FinvT=np.linalg.inv(F).T; I1b=J**(-2/3)*I1; I2b=J**(-4/3)*I2
    Q1=J**(-2/3)*(F-(I1/3.)*FinvT); D2=J**(-4/3)*(2*(I1*F-F@C)-(4/3.)*I2*FinvT)
    h95=hfun(I1b,I2b,.95)
    if kind==0:
        r1=c*2*h95; r2=c*(-2*.95*h95)
    else:
        h85=hfun(I1b,I2b,.85); r1=c*(2*h95*h85*h85+2*h85*h95*h95); r2=c*(-2*.95*h95*h85*h85-2*.85*h85*h95*h95)
    return 2*r1*Q1+r2*D2

def top_stress(y,gamma,kind,c):
    coords,p=m.unpack(mesh,y,gamma,EPS); out={'r_cur':[],'szz_Pa':[]}; G0=float(m.G_KPA)
    for i in range(mesh['nr']):
        ei=(mesh['nz']-1)*mesh['nr']+i; uid=mesh['eu'][ei]; pid=mesh['ep'][ei]; Xe=mesh['X'][uid]; ve=coords[uid]; pe=p[pid]; eta=1.; le,dle=m.L2(eta); lp_e,_=m.L1(eta)
        for xi in m.GP3:
            lx,dlx=m.L2(xi); lp_x,_=m.L1(xi); N=[]; dxi=[]; deta=[]
            for bb in range(3):
                for aa in range(3): N.append(lx[aa]*le[bb]); dxi.append(dlx[aa]*le[bb]); deta.append(lx[aa]*dle[bb])
            N=np.asarray(N); dxi=np.asarray(dxi); deta=np.asarray(deta); J2=np.array([[dxi@Xe[:,0],deta@Xe[:,0]],[dxi@Xe[:,1],deta@Xe[:,1]]]); DG=np.column_stack([dxi,deta])@np.linalg.inv(J2); val=N@ve; der=np.einsum('ia,ic->ca',DG,ve); rr=val[0]; Rq=float(N@Xe[:,0]); F=np.array([[der[0,0],0.,der[0,1]],[rr*der[1,0],rr/Rq,rr*der[1,1]],[der[2,0],0.,der[2,1]]]); J=np.linalg.det(F)
            Npv=np.array([lp_x[0]*lp_e[0],lp_x[1]*lp_e[0],lp_x[1]*lp_e[1],lp_x[0]*lp_e[1]]); pq=Npv@pe; P=post.piso(F,post.F_MR,0.)+perturb_P(F,kind,c)+pq*np.linalg.inv(F).T; sig=(P@F.T)/J*G0*1000.; out['r_cur'].append(rr); out['szz_Pa'].append(sig[2,2])
    return {k:np.asarray(v) for k,v in out.items()}
def raw(yb_,ys_,kind,c):
    p0=top_stress(yb_,0.,kind,c); pg=top_stress(ys_,GAM,kind,c); coords,_=m.unpack(mesh,ys_,GAM,EPS); top=np.isclose(mesh['X'][:,1],mesh['H']); a=float(np.max(coords[top,0])); o=np.argsort(p0['r_cur']); bp=np.interp(pg['r_cur'],p0['r_cur'][o],p0['szz_Pa'][o]); return pg['r_cur']/a,-(pg['szz_Pa']-bp),a
j=np.arange(25.); xs=np.sqrt(.05**2+(j+.5)/25*(.85**2-.05**2))
def sample(x,v):
    o=np.argsort(x); return np.interp(xs,np.asarray(x)[o],np.asarray(v)[o])
def tangent_scaled(kind,scale):
    sb=assemble_source(yb,True,kind); ss=assemble_source(ys,False,kind); dyb=scale*spla.spsolve(Hb,-sb); dys=scale*spla.spsolve(Hs,-ss); vals={}
    for h in [1e-4,5e-5,2.5e-5]:
        xp,pp,ap=raw(yb+h*dyb,ys+h*dys,kind,+h*scale); xm,pm,am=raw(yb-h*dyb,ys-h*dys,kind,-h*scale); vals[h]=((sample(xp,pp)-sample(xm,pm))/(2*h),(ap-am)/(2*h))
    def aang(a,b): return math.degrees(math.acos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))
    return vals[2.5e-5][0],float(vals[2.5e-5][1]),{'1e-4_vs_5e-5':aang(vals[1e-4][0],vals[5e-5][0]),'5e-5_vs_2.5e-5':aang(vals[5e-5][0],vals[2.5e-5][0])}
# Homogeneous stress normalization. Reproduce the single-zero coefficient and compute double-zero coefficient under identical sampled Frobenius stress-increment rule.
def Fhom(lam,gam): return np.array([[lam**-.5,0,0],[0,lam**-.5,gam*lam],[0,0,lam]],float)
def stress_norm(kind,lam,gam):
    F=Fhom(lam,gam); return float(np.linalg.norm((perturb_P(F,kind,1.)@F.T),ord='fro'))
norms={}
for kind,name in [(0,'single_0.95'),(1,'double_0.95_0.85')]:
    mx=(-1,None)
    for lam in np.linspace(.90,1.00,101):
        for gam in np.linspace(0.,.8,161):
            n=stress_norm(kind,float(lam),float(gam))
            if n>mx[0]: mx=(n,(float(lam),float(gam)))
    norms[name]={'unit_max_stress_increment_over_G':mx[0],'argmax_lambda_gamma':mx[1],'coefficient_for_0.05G':.05/mx[0]}
c1=norms['single_0.95']['coefficient_for_0.05G']; cd=norms['double_0.95_0.85']['coefficient_for_0.05G']; G1,da1,step1=tangent_scaled(0,c1); GD,dad,stepd=tangent_scaled(1,cd)
# pressure fit coefficients and derived normal-stress coefficients
def fit(g):
    X=np.c_[np.ones_like(xs),xs*xs]; A,B=np.linalg.lstsq(X,g,rcond=None)[0]; _,_,a=raw(yb,ys,0,0.); gammaobs=GAM*a/m.R; dq=-(A+B)/(gammaobs**2)/1000.; dalpha=(3*A+B)/(gammaobs**2)/1000.; return {'A_Pa':float(A),'B_Pa':float(B),'delta_q_kPa':float(dq),'delta_alpha1_kPa':float(dalpha),'gammaobs':float(gammaobs)}
out={'schema':'U1-R196-double-preload-null-control-v1','normalization':norms,'coefficient_ratio_double_over_single':cd/c1,'primary_observer':'25 equal-area x=0.05..0.85','single_zero':{'profile_norm_Pa':float(np.linalg.norm(G1)),'profile_norm_SD':float(np.linalg.norm(G1)/SIGMA),'radius_tangent_scaled':da1,'step_angle_deg':step1,'fit':fit(G1)},'double_zero':{'profile_norm_Pa':float(np.linalg.norm(GD)),'profile_norm_SD':float(np.linalg.norm(GD)/SIGMA),'radius_tangent_scaled':dad,'step_angle_deg':stepd,'fit':fit(GD)},'double_over_single_profile_norm':float(np.linalg.norm(GD)/np.linalg.norm(G1)),'equilibrium_grad_inf_pre':float(np.linalg.norm(gb,np.inf)),'equilibrium_grad_inf_twist':float(np.linalg.norm(gs,np.inf))}
(OUT/'R196_DOUBLE_PRELOAD_NULL_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n'); np.savetxt(OUT/'R196_PRESSURE_TANGENTS.csv',np.column_stack([xs,G1,GD]),delimiter=',',header='x,single_zero_scaled_Pa,double_zero_scaled_Pa',comments=''); print(json.dumps(out,indent=2))
