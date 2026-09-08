#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import importlib.util, json, math, sys, time
import numpy as np
import scipy.sparse.linalg as spla
import jax, jax.numpy as jnp

HERE=Path(__file__).resolve().parent
INP=HERE/'input'; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(INP))
import postprocess_core as post

K=10000.; EPS=.05; GAM=.2; BETA=np.array([1000.,3000.,3000.])
NR=34; NZ=10; ASPECT=10.; EDGE_CELLS=10; EDGE_WIDTH=.1
CAMP=0.764412; SIGMA=2.0
OLD="""    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    Wi=.5*(Js**(-2/3)*I1-3.)\n"""
MR="""    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    C2=C@C; I2=.5*(I1*I1-jnp.trace(C2,axis1=-2,axis2=-1))\n    f=0.30*0.95/(1.0-0.30*(1.0-0.95))\n    I1b=Js**(-2/3)*I1; I2b=Js**(-4/3)*I2\n    Wi=.5*(1-f)*(I1b-3.)+.5*f*(I2b-3.)\n"""

def load_mr():
    src=(INP/'solver_core.py').read_text(); tmp=OUT/'_r208_mr.py'; tmp.write_text(src.replace(OLD,MR))
    sp=importlib.util.spec_from_file_location('r208_mr',tmp); m=importlib.util.module_from_spec(sp); assert sp.loader; sp.loader.exec_module(m); return m
m=load_mr()

def points():
    j=np.arange(25.); return np.sqrt(.05**2+(j+.5)/25*(.85**2-.05**2))
XS=points()

def sample(x,v,xs=XS):
    o=np.argsort(x); return np.interp(xs,np.asarray(x)[o],np.asarray(v)[o])

def angle(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na==0 or nb==0: return float('nan')
    return math.degrees(math.acos(float(np.clip(np.dot(a,b)/(na*nb),-1,1))))

def src_energy(z,N,DG,Rq,W,mask):
    ve=z[:27].reshape(9,3); val=jnp.einsum('gi,ic->gc',N,ve); der=jnp.einsum('gia,ic->gca',DG,ve)
    rr=val[:,0]; rR=der[:,0,0]; rz=der[:,0,1]; pR=der[:,1,0]; pz=der[:,1,1]; zR=der[:,2,0]; zz=der[:,2,1]; zero=jnp.zeros_like(rr)
    F=jnp.stack([jnp.stack([rR,zero,rz],1),jnp.stack([rr*pR,rr/Rq,rr*pz],1),jnp.stack([zR,zero,zz],1)],1)
    J=jnp.linalg.det(F); Js=jnp.maximum(J,1e-10); C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1); C2=C@C; I2=.5*(I1*I1-jnp.trace(C2,axis1=-2,axis2=-1))
    lam=.95; I10=lam*lam+2/lam; I20=2*lam+lam**-2; h=Js**(-2/3)*I1-I10-lam*(Js**(-4/3)*I2-I20)
    return jnp.sum(mask*h*h*W)
src_grad=jax.jit(jax.vmap(jax.grad(src_energy),in_axes=(0,0,0,0,0,0)))

def assemble_source(mesh,y,mask,is_pre):
    lv=m.local_vars(mesh,y,0.,EPS) if is_pre else m.local_vars(mesh,y,GAM,EPS)
    gl=np.asarray(src_grad(jnp.asarray(lv),jnp.asarray(mesh['N']),jnp.asarray(mesh['G']),jnp.asarray(mesh['Rq']),jnp.asarray(mesh['W']),jnp.asarray(mask)))
    gg=np.zeros(len(y))
    for e,lm in enumerate(mesh['maps']):
        ok=lm>=0; np.add.at(gg,lm[ok],gl[e,ok])
    return gg

def beta_residual_derivative(mesh,y,gamma,channel):
    coords,_=m.unpack(mesh,y,gamma,EPS); gg=np.zeros(len(y)); H=mesh['H']; theta=gamma*H*(1-EPS)/m.R
    kbase=BETA/H
    for ids,smaps,Ns,rr,wt,tap,zidx in mesh['surf']:
        c=int(channel); cur=np.array([float(Ns@coords[ids,q]) for q in range(3)]); is_top=(zidx==2*mesh['nz'])
        target=np.array([rr,theta if is_top else 0.,H*(1-EPS) if is_top else 0.]); delta=cur[c]-target[c]
        scales=np.array([1.,rr,1.]); kk=kbase[c]*wt*tap*scales[c]**2
        smap=smaps[c]; active=np.flatnonzero(smap>=0); gm=smap[active]; Na=Ns[active]
        np.add.at(gg,gm,kk*delta*Na)
    return gg

def top_stress(mesh,y,gamma,c):
    coords,p=m.unpack(mesh,y,gamma,EPS); out={k:[] for k in ['r_cur','szz_Pa']}; G0=float(m.G_KPA)
    for i in range(mesh['nr']):
        ei=(mesh['nz']-1)*mesh['nr']+i; uid=mesh['eu'][ei]; pid=mesh['ep'][ei]; Xe=mesh['X'][uid]; ve=coords[uid]; pe=p[pid]; eta=1.; le,dle=m.L2(eta); lp_e,_=m.L1(eta)
        for xi in m.GP3:
            lx,dlx=m.L2(xi); lp_x,_=m.L1(xi); N=[]; dxi=[]; deta=[]
            for bb in range(3):
                for aa in range(3): N.append(lx[aa]*le[bb]); dxi.append(dlx[aa]*le[bb]); deta.append(lx[aa]*dle[bb])
            N=np.asarray(N); dxi=np.asarray(dxi); deta=np.asarray(deta); J2=np.array([[dxi@Xe[:,0],deta@Xe[:,0]],[dxi@Xe[:,1],deta@Xe[:,1]]]); DG=np.column_stack([dxi,deta])@np.linalg.inv(J2)
            val=N@ve; der=np.einsum('ia,ic->ca',DG,ve); rr=val[0]; Rq=float(N@Xe[:,0]); F=np.array([[der[0,0],0.,der[0,1]],[rr*der[1,0],rr/Rq,rr*der[1,1]],[der[2,0],0.,der[2,1]]]); J=np.linalg.det(F)
            Npv=np.array([lp_x[0]*lp_e[0],lp_x[1]*lp_e[0],lp_x[1]*lp_e[1],lp_x[0]*lp_e[1]]); pq=Npv@pe
            P=post.piso(F,post.F_MR,c,0.,0.)+pq*np.linalg.inv(F).T; sig=(P@F.T)/J*G0*1000.; out['r_cur'].append(rr); out['szz_Pa'].append(sig[2,2])
    return {k:np.asarray(v) for k,v in out.items()}

def raw_prof(mesh,ybp,ysp,c=0.):
    p0=top_stress(mesh,ybp,0.,c); pg=top_stress(mesh,ysp,GAM,c); coords,_=m.unpack(mesh,ysp,GAM,EPS); top=np.isclose(mesh['X'][:,1],mesh['H']); a=float(np.max(coords[top,0])); o=np.argsort(p0['r_cur']); bp=np.interp(pg['r_cur'],p0['r_cur'][o],p0['szz_Pa'][o]); return pg['r_cur']/a,-(pg['szz_Pa']-bp),a

def tangent_profile(mesh,yb,ys,dyb,dys,direct_c=0.):
    vals={}
    for h in [1e-4,5e-5]:
        xp,pp,ap=raw_prof(mesh,yb+h*dyb,ys+h*dys,+h*direct_c); xm,pm,am=raw_prof(mesh,yb-h*dyb,ys-h*dys,-h*direct_c)
        vals[h]=((sample(xp,pp)-sample(xm,pm))/(2*h),(ap-am)/(2*h))
    return vals[5e-5][0], angle(vals[1e-4][0],vals[5e-5][0]), float(vals[5e-5][1])

def continuation(mesh):
    # Direct target-state solve. The thin-gap homogeneous geometry provides a close initial state,
    # so intermediate continuation increments are unnecessary for this gate.
    y0=m.initial(mesh,0.,EPS); _=m.assemble(mesh,y0,K,0.,EPS,True,*BETA)
    yb,okb,hb=m.newton(mesh,y0,K,0.,EPS,*BETA,tol=1e-8,maxit=50)
    if not okb: raise RuntimeError(f'pre failed {hb[-1]}')
    ys,oks,hs=m.newton(mesh,yb.copy(),K,GAM,EPS,*BETA,tol=1e-8,maxit=50)
    if not oks: raise RuntimeError(f'twist failed {hs[-1]}')
    return yb,ys,hb+hs

def main():
    t0=time.time(); prereg=json.loads((HERE/'R208_PREREGISTRATION.json').read_text())
    mesh=m.build(NR,NZ,ASPECT,EDGE_WIDTH,EDGE_CELLS,None,'tapered_springs')
    yb,ys,hist=continuation(mesh)
    np.savez_compressed(OUT/'baseline_aspect10_34x10.npz',yb=yb,ys=ys,rp=mesh['rp'],H=mesh['H'])
    Eb,gb,Hb=m.assemble(mesh,yb,K,0.,EPS,True,*BETA); Es,gs,Hs=m.assemble(mesh,ys,K,GAM,EPS,True,*BETA)
    cols=[]; beta_checks=[]
    for c in range(3):
        sb=beta_residual_derivative(mesh,yb,0.,c); ss=beta_residual_derivative(mesh,ys,GAM,c); dyb=spla.spsolve(Hb,-sb); dys=spla.spsolve(Hs,-ss)
        g,sa,rd=tangent_profile(mesh,yb,ys,dyb,dys,0.); cols.append(g); beta_checks.append({'channel':c,'step_angle_deg':sa,'radius_deriv':rd})
    B=np.column_stack(cols); U,sv,_=np.linalg.svd(B,full_matrices=False)
    mask=np.ones_like(np.asarray(mesh['Rq'])); sb=assemble_source(mesh,yb,mask,True); ss=assemble_source(mesh,ys,mask,False); dyb=spla.spsolve(Hb,-sb); dys=spla.spsolve(Hs,-ss); g,sa,rd=tangent_profile(mesh,yb,ys,dyb,dys,1.)
    orth=g-U@(U.T@g); frac=float(np.linalg.norm(orth)/np.linalg.norm(g)); norm_pa=float(np.linalg.norm(g)); response_sd=float(CAMP*norm_pa/SIGMA)
    base_ref=json.load(open(INP/'reference_HR0p20_case.json'))
    ref_norm=float(base_ref['full']['norm']); ref_frac=float(base_ref['full']['orth_fraction'])
    valid=bool(np.linalg.norm(gb,np.inf)<prereg['pass_rules']['baseline_residual_inf_max'] and np.linalg.norm(gs,np.inf)<prereg['pass_rules']['baseline_residual_inf_max'] and max([sa]+[x['step_angle_deg'] for x in beta_checks])<prereg['pass_rules']['tangent_step_angle_deg_max'])
    pass_response=response_sd>=prereg['pass_rules']['finite_response_SD_min']; pass_overlap=frac<=prereg['pass_rules']['outside_interface_span_fraction_max']
    verdict='THIN_GAP_RELEVANCE_SUPPORTED' if valid and pass_response and pass_overlap else ('INVALID_NUMERICAL_GATE' if not valid else 'THIN_GAP_RELEVANCE_NOT_SUPPORTED')
    out={
      'schema':'U1-R208-thin-gap-relevance-result-v1','verdict':verdict,'preregistration_sha256':(HERE/'R208_PREREGISTRATION.sha256').read_text().split()[0],
      'geometry':{'H0_over_R0':0.1,'aspect_R0_over_H0':10.0,'mesh':'34x10_ec10','reference_axial_element_size':mesh['H']/NZ},
      'baseline':{'grad_inf_pre':float(np.linalg.norm(gb,np.inf)),'grad_inf_twist':float(np.linalg.norm(gs,np.inf)),'runtime_s':time.time()-t0},
      'interface_singular_values_Pa_per_logbeta':sv.tolist(),'interface_step_checks':beta_checks,
      'constitutive_tangent':{'norm_Pa_per_c':norm_pa,'published_amplitude_response_SD':response_sd,'outside_interface_span_norm_Pa_per_c':float(np.linalg.norm(orth)),'outside_interface_span_fraction':frac,'angle_to_interface_span_deg':float(math.degrees(math.asin(np.clip(frac,0,1)))),'step_angle_deg':sa,'radius_deriv':rd},
      'reference_H_over_R_0p20':{'norm_Pa_per_c':ref_norm,'outside_interface_span_fraction':ref_frac,'thin_to_reference_norm_ratio':norm_pa/ref_norm},
      'pass_flags':{'numerical_validity':valid,'finite_response_SD':pass_response,'interface_span_overlap':pass_overlap}
    }
    np.savetxt(OUT/'R208_PRESSURE_TANGENTS.csv',np.column_stack([XS,cols[0],cols[1],cols[2],g]),delimiter=',',header='x,beta_r_log,beta_theta_log,beta_z_log,constitutive_Q1',comments='')
    (OUT/'R208_THIN_GAP_RELEVANCE_RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
