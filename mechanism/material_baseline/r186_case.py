#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, importlib.util, json, math, sys
import numpy as np
import scipy.sparse.linalg as spla
import jax, jax.numpy as jnp

HERE=Path(__file__).resolve().parent; INP=HERE/'input'; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(INP)); import postprocess_core as post
K=10000.; EPS=.05; GAM=.20; BETA=np.array([1000.,3000.,3000.]); NR=34; NZ=20; EDGE_CELLS=10
OLD="""    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    Wi=.5*(Js**(-2/3)*I1-3.)\n"""

def load_mr(fbase: float):
    MR=f"""    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    C2=C@C; I2=.5*(I1*I1-jnp.trace(C2,axis1=-2,axis2=-1))\n    f={fbase:.17g}\n    I1b=Js**(-2/3)*I1; I2b=Js**(-4/3)*I2\n    Wi=.5*(1-f)*(I1b-3.)+.5*f*(I2b-3.)\n"""
    src=(INP/'solver_core.py').read_text()
    if OLD not in src: raise RuntimeError('solver material block not found')
    tmp=OUT/f'_r186_mr_f{fbase:.5f}.py'; tmp.write_text(src.replace(OLD,MR))
    name='r186_mr_'+str(fbase).replace('.','p')
    sp=importlib.util.spec_from_file_location(name,tmp); mod=importlib.util.module_from_spec(sp); assert sp.loader; sp.loader.exec_module(mod); return mod

def angle(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na==0 or nb==0: return float('nan')
    return math.degrees(math.acos(float(np.clip(np.dot(a,b)/(na*nb),-1,1))))

def smooth_inner_weight(r,center,halfwidth,R=1.0):
    r=np.asarray(r,float)/float(R); c=float(center); d=float(halfwidth); lo,hi=c-d,c+d
    t=np.clip((r-lo)/(hi-lo),0.,1.); s=6*t**5-15*t**4+10*t**3; w=1-s
    return np.where(r<=lo,1.,np.where(r>=hi,0.,w))

def run(fbase: float):
    m=load_mr(fbase)
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
        coords,_=m.unpack(mesh,y,gamma,EPS); gg=np.zeros(len(y)); H=mesh['H']; theta=gamma*H*(1-EPS)/m.R; kbase=BETA/H
        for ids,smaps,Ns,rr,wt,tap,zidx in mesh['surf']:
            c=int(channel); cur=np.array([float(Ns@coords[ids,q]) for q in range(3)]); is_top=(zidx==2*mesh['nz'])
            target=np.array([rr,theta if is_top else 0.,H*(1-EPS) if is_top else 0.]); delta=cur[c]-target[c]; scales=np.array([1.,rr,1.]); kk=kbase[c]*wt*tap*scales[c]**2
            smap=smaps[c]; active=np.flatnonzero(smap>=0); gm=smap[active]; Na=Ns[active]; np.add.at(gg,gm,kk*delta*Na)
        return gg
    def top_stress_weighted(mesh,y,gamma,c,center,halfwidth):
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
                w=1.0 if center is None else float(smooth_inner_weight(np.array([Rq]),center,halfwidth,m.R)[0])
                P=post.piso(F,fbase,c*w,0.,0.)+pq*np.linalg.inv(F).T; sig=(P@F.T)/J*G0*1000.; out['r_cur'].append(rr); out['szz_Pa'].append(sig[2,2])
        return {k:np.asarray(v) for k,v in out.items()}
    def raw_prof(mesh,ybp,ysp,c=0.,center=None,halfwidth=.02):
        p0=top_stress_weighted(mesh,ybp,0.,c,center,halfwidth); pg=top_stress_weighted(mesh,ysp,GAM,c,center,halfwidth); coords,_=m.unpack(mesh,ysp,GAM,EPS); top=np.isclose(mesh['X'][:,1],mesh['H']); a=float(np.max(coords[top,0])); o=np.argsort(p0['r_cur']); bp=np.interp(pg['r_cur'],p0['r_cur'][o],p0['szz_Pa'][o]); return pg['r_cur']/a,-(pg['szz_Pa']-bp),a
    xs=lambda: np.sqrt(.05**2+(np.arange(25.)+.5)/25*(.80**2-.05**2))
    XOBS=xs()
    def samp(x,v):
        o=np.argsort(x); return np.interp(XOBS,np.asarray(x)[o],np.asarray(v)[o])
    def prof_deriv(mesh,yb,ys,dyb,dys,center,kind):
        vals={}
        for h in [1e-4,5e-5]:
            if kind=='full': plus=(yb+h*dyb,ys+h*dys,+h); minus=(yb-h*dyb,ys-h*dys,-h)
            elif kind=='direct': plus=(yb,ys,+h); minus=(yb,ys,-h)
            elif kind=='state': plus=(yb+h*dyb,ys+h*dys,0.); minus=(yb-h*dyb,ys-h*dys,0.)
            else: raise ValueError(kind)
            xp,pp,_=raw_prof(mesh,plus[0],plus[1],plus[2],center,.02); xm,pm,_=raw_prof(mesh,minus[0],minus[1],minus[2],center,.02)
            vals[h]=(samp(xp,pp)-samp(xm,pm))/(2*h)
        return vals[5e-5],angle(vals[1e-4],vals[5e-5])
    def beta_prof(mesh,yb,ys,db,ds):
        vals={}
        for h in [1e-4,5e-5]:
            xp,pp,_=raw_prof(mesh,yb+h*db,ys+h*ds,0.,None,.02); xm,pm,_=raw_prof(mesh,yb-h*db,ys-h*ds,0.,None,.02)
            vals[h]=(samp(xp,pp)-samp(xm,pm))/(2*h)
        return vals[5e-5],angle(vals[1e-4],vals[5e-5])

    mesh=m.build(NR,NZ,5.,.1,EDGE_CELLS,None,'tapered_springs')
    ref=np.load(INP/'reference_state_f0.289340.npz'); yb0=np.asarray(ref['yb']); ys0=np.asarray(ref['ys'])
    # Re-equilibrate both endpoints under the new material. The historical states are initial guesses only.
    yb,ok,hb=m.newton(mesh,yb0.copy(),K,0.,EPS,*BETA,tol=1e-8,maxit=60)
    if not ok: raise RuntimeError(f'preload equilibrium failed for f={fbase}: {hb[-1]}')
    ys,ok,hs=m.newton(mesh,ys0.copy(),K,GAM,EPS,*BETA,tol=1e-8,maxit=60)
    if not ok:
        # Fixed continuation fallback; not result-adaptive.
        y=yb.copy(); hist=[]; ok=True
        for gg in [.05,.10,.15,.20]:
            y,oo,hh=m.newton(mesh,y,K,gg,EPS,*BETA,tol=1e-8,maxit=60); hist+=hh
            if not oo: ok=False; break
        ys=y; hs=hist
    if not ok: raise RuntimeError(f'twist equilibrium failed for f={fbase}: {hs[-1]}')
    np.savez_compressed(OUT/f'state_f_{fbase:.2f}.npz',yb=yb,ys=ys,f=fbase)
    _,gb,Hb=m.assemble(mesh,yb,K,0.,EPS,True,*BETA); _,gs,Hs=m.assemble(mesh,ys,K,GAM,EPS,True,*BETA)
    cols=[]; maxstep=0.
    for ch in range(3):
        sb=beta_residual_derivative(mesh,yb,0.,ch); ss=beta_residual_derivative(mesh,ys,GAM,ch); db=spla.spsolve(Hb,-sb); ds=spla.spsolve(Hs,-ss); g,sa=beta_prof(mesh,yb,ys,db,ds); cols.append(g); maxstep=max(maxstep,sa)
    U,sv,_=np.linalg.svd(np.column_stack(cols),full_matrices=False); orth=lambda g:g-U@(U.T@g)
    data={}
    for center in [.77,.83]:
        mask=smooth_inner_weight(np.asarray(mesh['Rq']),center,.02,m.R); sb=assemble_source(mesh,yb,mask,True); ss=assemble_source(mesh,ys,mask,False); db=spla.spsolve(Hb,-sb); ds=spla.spsolve(Hs,-ss); data[center]={}
        for kind in ['full','direct','state']:
            g,sa=prof_deriv(mesh,yb,ys,db,ds,center,kind); data[center][kind]=g; maxstep=max(maxstep,sa)
    inner=orth(data[.77]['full']); bands={k:orth(data[.83][k]-data[.77][k]) for k in ['full','direct','state']}
    nd=np.linalg.norm(bands['direct']); ns=np.linalg.norm(bands['state']); nf=np.linalg.norm(bands['full']); ni=np.linalg.norm(inner); after=np.linalg.norm(inner+bands['full'])
    dsang=angle(bands['direct'],bands['state']); ratio=float(nd/ns); resid_ratio=float(nf/max(nd,ns)); closure=float(np.linalg.norm(bands['full']-bands['direct']-bands['state'])/max(nf,1e-30)); l1=bool(dsang>160 and .75<=ratio<=1.33 and resid_ratio<=.35)
    totalang=angle(inner,bands['full']); reduction=1-float(after/ni); l2=bool(totalang>150 and reduction>=.5)
    valid=bool(maxstep<.01 and closure<1e-5 and np.linalg.norm(gb,np.inf)<1e-7 and np.linalg.norm(gs,np.inf)<1e-7)
    rec={'schema':'U1-R186-material-case-v1','f':fbase,'small_strain_shear_normalization':'fixed coefficient sum','equilibrium_grad_inf_pre':float(np.linalg.norm(gb,np.inf)),'equilibrium_grad_inf_twist':float(np.linalg.norm(gs,np.inf)),'interface_singular_values':sv.tolist(),'max_step_angle_deg':float(maxstep),'level1':{'direct_state_angle_deg':float(dsang),'direct_orth_norm':float(nd),'state_orth_norm':float(ns),'direct_over_state':ratio,'total_band_orth_norm':float(nf),'total_over_max_component':resid_ratio,'closure_rel':closure,'pass':l1},'level2':{'inner_orth_norm':float(ni),'total_band_angle_to_inner_deg':float(totalang),'orth_norm_after':float(after),'reduction_fraction':float(reduction),'pass':l2},'valid':valid,'pass':bool(valid and l1 and l2)}
    (OUT/f'case_f_{fbase:.2f}.json').write_text(json.dumps(rec,indent=2)+'\n'); print(json.dumps(rec,indent=2),flush=True); return rec

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--f',type=float,required=True); a=ap.parse_args(); run(a.f)
