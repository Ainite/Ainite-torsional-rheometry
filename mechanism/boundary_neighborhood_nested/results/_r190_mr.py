from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

R=1.0
G_KPA=23/(2*(1+.4))
GP3=np.array([-math.sqrt(3/5),0.,math.sqrt(3/5)])
GW3=np.array([5/9,8/9,5/9])

def L2(x):
    return np.array([.5*x*(x-1),1-x*x,.5*x*(x+1)]), np.array([x-.5,-2*x,x+.5])
def L1(x):
    return np.array([(1-x)/2,(1+x)/2]),np.array([-.5,.5])

def radial_knots(nr, edge_width=.1, edge_cells=None, grading_power=None):
    if grading_power is not None:
        ss=np.linspace(0.,1.,nr+1); return R*(1.-(1.-ss)**float(grading_power))
    if edge_cells is None: edge_cells=max(4,int(round(.30*nr)))
    edge_cells=min(edge_cells,nr-2); bulk=nr-edge_cells
    r0=R*(1-edge_width)
    return np.r_[np.linspace(0,r0,bulk+1),np.linspace(r0,R,edge_cells+1)[1:]]

def q2_from_coarse(c):
    out=[]
    for i in range(len(c)-1): out += [c[i],.5*(c[i]+c[i+1])]
    out.append(c[-1]); return np.asarray(out)

def build(nr,nz,aspect=5.,edge_width=.1,edge_cells=None,grading_power=None,plate_mode='tapered_springs'):
    H=R/aspect
    rp=radial_knots(nr,edge_width,edge_cells,grading_power)
    zp=np.linspace(0,H,nz+1)
    r2=q2_from_coarse(rp); z2=q2_from_coarse(zp)
    X=np.array([[r,z] for z in z2 for r in r2],float); n2r=len(r2)
    Xp=np.array([[r,z] for z in zp for r in rp],float); npr=len(rp)
    eu=[]; ep=[]
    for j in range(nz):
        for i in range(nr):
            eu.append([(2*j+b)*n2r+(2*i+a) for b in range(3) for a in range(3)])
            ep.append([j*npr+i,j*npr+i+1,(j+1)*npr+i+1,(j+1)*npr+i])
    eu=np.asarray(eu,int); ep=np.asarray(ep,int)
    nd=len(X); fixed=np.zeros((nd,3),bool)
    for n,(rr,zz) in enumerate(X):
        onplate=abs(zz)<1e-12 or abs(zz-H)<1e-12
        if onplate and plate_mode=='dirichlet':
            fixed[n,1]=True; fixed[n,2]=True
        elif onplate and plate_mode=='bonded':
            fixed[n,:]=True
        if abs(rr)<1e-12:
            fixed[n,0]=True; fixed[n,1]=True
    free_flat=np.flatnonzero(~fixed.ravel())
    free_index=-np.ones(nd*3,dtype=int); free_index[free_flat]=np.arange(len(free_flat))
    # Element quadrature arrays
    Ns=[]; Gs=[]; Nps=[]; Rqs=[]; Ws=[]
    for ee in range(len(eu)):
        Xe=X[eu[ee]]; ens=[]; egs=[]; enp=[]; er=[]; ew=[]
        for ib,eta in enumerate(GP3):
            le,dle=L2(eta); lp_e,_=L1(eta)
            for ia,xi in enumerate(GP3):
                lx,dlx=L2(xi); lp_x,_=L1(xi)
                N=[]; dxi=[]; deta=[]
                for b in range(3):
                    for a in range(3):
                        N.append(lx[a]*le[b]); dxi.append(dlx[a]*le[b]); deta.append(lx[a]*dle[b])
                N=np.asarray(N); dxi=np.asarray(dxi); deta=np.asarray(deta)
                J2=np.array([[dxi@Xe[:,0],deta@Xe[:,0]],[dxi@Xe[:,1],deta@Xe[:,1]]])
                DG=np.column_stack([dxi,deta])@np.linalg.inv(J2)
                rr=N@Xe[:,0]; wt=np.linalg.det(J2)*2*math.pi*rr*GW3[ia]*GW3[ib]
                Np=np.array([lp_x[0]*lp_e[0],lp_x[1]*lp_e[0],lp_x[1]*lp_e[1],lp_x[0]*lp_e[1]])
                ens.append(N); egs.append(DG); enp.append(Np); er.append(rr); ew.append(wt)
        Ns.append(ens); Gs.append(egs); Nps.append(enp); Rqs.append(er); Ws.append(ew)
    # local-to-global map: 27 displacement components + 4 pressure nodes
    nf=len(free_flat); maps=[]
    for ue,pe in zip(eu,ep):
        lm=[]
        for nid in ue:
            for c in range(3): lm.append(free_index[3*nid+c])
        lm += [nf+int(pid) for pid in pe]
        maps.append(lm)
    maps=np.asarray(maps,int)
    # surface quadrature for radial spring
    surf=[]
    for zidx in [0,2*nz]:
        for i in range(nr):
            ids=np.array([zidx*n2r+2*i+a for a in range(3)])
            Xe=X[ids]
            for ia,xi in enumerate(GP3):
                lx,dlx=L2(xi); rr=lx@Xe[:,0]; dr=dlx@Xe[:,0]
                wt=2*math.pi*rr*dr*GW3[ia]
                x=(R-rr)/(edge_width*R) if edge_width>0 else 2.
                tap=0. if x<=0 else (1. if x>=1 else x*x*(3-2*x))
                smaps=np.array([[free_index[3*nid+c] for nid in ids] for c in range(3)],int)
                surf.append((ids,smaps,lx,rr,wt,tap,zidx))
    actual_edge=int(np.sum(rp[:-1] >= R*(1-edge_width)-1e-12))
    return dict(nr=nr,nz=nz,H=H,aspect=aspect,edge_width=edge_width,edge_cells=actual_edge,grading_power=grading_power,plate_mode=plate_mode,rp=rp,zp=zp,X=X,Xp=Xp,eu=eu,ep=ep,fixed=fixed,free_flat=free_flat,free_index=free_index,nf=nf,npn=len(Xp),N=np.asarray(Ns),G=np.asarray(Gs),Np=np.asarray(Nps),Rq=np.asarray(Rqs),W=np.asarray(Ws),maps=maps,surf=surf)

# Element potential accepts 31 local variables: 9x3 current coordinates + 4 pressure nodal values.
def _el_energy(z,N,DG,Np,Rq,W,K):
    ve=z[:27].reshape(9,3); pe=z[27:]
    val=jnp.einsum('gi,ic->gc',N,ve)
    der=jnp.einsum('gia,ic->gca',DG,ve)
    rr=val[:,0]; rR=der[:,0,0]; rz=der[:,0,1]; pR=der[:,1,0]; pz=der[:,1,1]; zR=der[:,2,0]; zz=der[:,2,1]
    zero=jnp.zeros_like(rr)
    F=jnp.stack([jnp.stack([rR,zero,rz],1),jnp.stack([rr*pR,rr/Rq,rr*pz],1),jnp.stack([zR,zero,zz],1)],1)
    J=jnp.linalg.det(F); Js=jnp.maximum(J,1e-10)
    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)
    C2=C@C; I2=.5*(I1*I1-jnp.trace(C2,axis1=-2,axis2=-1))
    f=0.30*0.95/(1.0-0.30*(1.0-0.95))
    I1b=Js**(-2/3)*I1; I2b=Js**(-4/3)*I2
    Wi=.5*(1-f)*(I1b-3.)+.5*f*(I2b-3.)
    pq=jnp.einsum('gi,i->g',Np,pe)
    wm=Wi+pq*jnp.log(Js)-.5*pq*pq/K
    barrier=1e7*jnp.square(jnp.maximum(.2-J,0.))
    return jnp.sum((wm+barrier)*W)

_el_valgrad=jax.jit(jax.vmap(jax.value_and_grad(_el_energy),in_axes=(0,0,0,0,0,0,None)))
_el_hess=jax.jit(jax.vmap(jax.hessian(_el_energy),in_axes=(0,0,0,0,0,0,None)))

def base_coords(mesh,gamma,eps):
    H=mesh['H']; theta=gamma*H*(1-eps)/R; X=mesh['X']
    return np.column_stack([X[:,0],theta*X[:,1]/H,(1-eps)*X[:,1]])

def unpack(mesh,y,gamma,eps):
    coords=base_coords(mesh,gamma,eps).reshape(-1)
    coords[mesh['free_flat']]=y[:mesh['nf']]
    return coords.reshape(-1,3), y[mesh['nf']:]

def initial(mesh,gamma=0.,eps=0.):
    c=base_coords(mesh,gamma,eps).reshape(-1)
    return np.r_[c[mesh['free_flat']],np.zeros(mesh['npn'])]

def local_vars(mesh,y,gamma,eps):
    coords,p=unpack(mesh,y,gamma,eps)
    return np.concatenate([coords[mesh['eu']].reshape(len(mesh['eu']),27),p[mesh['ep']]],axis=1)

def assemble(mesh,y,K,gamma,eps,need_hess=True,beta_r=1000.,beta_theta=None,beta_z=None):
    lv=local_vars(mesh,y,gamma,eps)
    vals,grads=_el_valgrad(jnp.asarray(lv),jnp.asarray(mesh['N']),jnp.asarray(mesh['G']),jnp.asarray(mesh['Np']),jnp.asarray(mesh['Rq']),jnp.asarray(mesh['W']),float(K))
    grads=np.asarray(grads); ndof=len(y); gg=np.zeros(ndof)
    maps=mesh['maps']
    for e,lm in enumerate(maps):
        mask=lm>=0; np.add.at(gg,lm[mask],grads[e,mask])
    rows=[]; cols=[]; data=[]
    if need_hess:
        Hloc=np.asarray(_el_hess(jnp.asarray(lv),jnp.asarray(mesh['N']),jnp.asarray(mesh['G']),jnp.asarray(mesh['Np']),jnp.asarray(mesh['Rq']),jnp.asarray(mesh['W']),float(K)))
        for e,lm in enumerate(maps):
            idx=np.flatnonzero(lm>=0); gm=lm[idx]; hh=Hloc[e][np.ix_(idx,idx)]
            rows.append(np.repeat(gm,len(gm))); cols.append(np.tile(gm,len(gm))); data.append(hh.ravel())
    # Consistently tapered plate springs in r, theta and z.  In plate_mode='dirichlet',
    # theta and z remain exact Dirichlet in the Dirichlet-interface configuration; otherwise all three
    # channels decay with the same C1 taper toward the free rim.
    coords,_=unpack(mesh,y,gamma,eps); Es=0.0
    if beta_theta is None: beta_theta=beta_r
    if beta_z is None: beta_z=beta_r
    kvec=np.array([beta_r,beta_theta,beta_z],float)/mesh['H']
    theta=gamma*mesh['H']*(1-eps)/R
    for ids,smaps,Ns,rr,wt,tap,zidx in mesh['surf']:
        cur=np.array([float(Ns@coords[ids,c]) for c in range(3)])
        is_top=(zidx==2*mesh['nz'])
        target=np.array([rr, theta if is_top else 0.0, mesh['H']*(1-eps) if is_top else 0.0])
        deltas=cur-target
        # theta coordinate is angular; convert mismatch to tangential displacement using reference radius.
        scales=np.array([1.0,rr,1.0])
        for c in range(3):
            # In Dirichlet-interface mode only the radial spring is active; theta/z are already fixed.
            if mesh.get('plate_mode')=='dirichlet' and c>0: continue
            if mesh.get('plate_mode')=='bonded': continue
            kk=kvec[c]*wt*tap*scales[c]**2
            Es += .5*kk*deltas[c]*deltas[c]
            smap=smaps[c]; active=np.flatnonzero(smap>=0); gm=smap[active]; Na=Ns[active]
            np.add.at(gg,gm,kk*deltas[c]*Na)
            if need_hess:
                hh=kk*np.outer(Na,Na); rows.append(np.repeat(gm,len(gm))); cols.append(np.tile(gm,len(gm))); data.append(hh.ravel())
    E=float(np.sum(np.asarray(vals)))+Es
    if not need_hess: return E,gg,None
    H=sp.coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(ndof,ndof)).tocsr()
    H.sum_duplicates()
    return E,gg,H

def newton(mesh,y0,K,gamma,eps,beta_r=1000.,beta_theta=None,beta_z=None,tol=1e-8,maxit=30):
    y=np.asarray(y0,float).copy(); hist=[]
    for it in range(maxit):
        E,g,H=assemble(mesh,y,K,gamma,eps,True,beta_r,beta_theta,beta_z); rn=float(np.linalg.norm(g,np.inf)); hist.append([rn,E])
        if rn<tol: return y,True,hist
        try: dy=spla.spsolve(H,-g)
        except Exception: dy=spla.lsmr(H,-g,atol=1e-12,btol=1e-12)[0]
        du=np.max(np.abs(dy[:mesh['nf']])) if mesh['nf'] else 0.
        if du>.04: dy*=.04/du
        phi=.5*np.dot(g,g); lam=1.; accepted=False
        for _ in range(20):
            yt=y+lam*dy; _,gt,_=assemble(mesh,yt,K,gamma,eps,False,beta_r,beta_theta,beta_z); ph=.5*np.dot(gt,gt)
            if np.isfinite(ph) and ph<phi:
                y=yt; accepted=True; break
            lam*=.5
        if not accepted: return y,False,hist
    return y,False,hist

def piso(F):
    J=np.linalg.det(F); C=F.T@F; I1=np.trace(C)
    # derivative of .5*J^(-2/3)*I1 wrt F = J^(-2/3)(F - I1/3 F^{-T})
    return J**(-2/3)*(F-(I1/3.)*np.linalg.inv(F).T)

def top_stress(mesh,y,K,gamma,eps):
    coords,p=unpack(mesh,y,gamma,eps); out={k:[] for k in ['r_ref','r_cur','a_cur','szz_Pa','srr_Pa','stt_Pa','tau_kPa']}
    for i in range(mesh['nr']):
        ei=(mesh['nz']-1)*mesh['nr']+i; uid=mesh['eu'][ei]; pid=mesh['ep'][ei]; Xe=mesh['X'][uid]; ve=coords[uid]; pe=p[pid]
        eta=1.; le,dle=L2(eta); lp_e,_=L1(eta)
        for ia,xi in enumerate(GP3):
            lx,dlx=L2(xi); lp_x,_=L1(xi); N=[]; dxi=[]; deta=[]
            for b in range(3):
                for a in range(3): N.append(lx[a]*le[b]); dxi.append(dlx[a]*le[b]); deta.append(lx[a]*dle[b])
            N=np.asarray(N); dxi=np.asarray(dxi); deta=np.asarray(deta)
            J2=np.array([[dxi@Xe[:,0],deta@Xe[:,0]],[dxi@Xe[:,1],deta@Xe[:,1]]]); DG=np.column_stack([dxi,deta])@np.linalg.inv(J2)
            val=N@ve; der=np.einsum('ia,ic->ca',DG,ve); rr=val[0]; Rq=N@Xe[:,0]
            F=np.array([[der[0,0],0,der[0,1]],[rr*der[1,0],rr/Rq,rr*der[1,1]],[der[2,0],0,der[2,1]]]); J=np.linalg.det(F)
            Npv=np.array([lp_x[0]*lp_e[0],lp_x[1]*lp_e[0],lp_x[1]*lp_e[1],lp_x[0]*lp_e[1]])
            pq=Npv@pe; P=piso(F)+pq*np.linalg.inv(F).T; sig=(P@F.T)/J*G_KPA*1000.
            dR=dxi@Xe[:,0]; drdR=der[0,0]; aw=2*math.pi*rr*drdR*dR*GW3[ia]
            out['r_ref'].append(Rq); out['r_cur'].append(rr); out['a_cur'].append(aw); out['szz_Pa'].append(sig[2,2]); out['srr_Pa'].append(sig[0,0]); out['stt_Pa'].append(sig[1,1]); out['tau_kPa'].append(sig[1,2]/1000.)
    return {k:np.asarray(v,float) for k,v in out.items()}

def jacobian_range(mesh,y,gamma,eps):
    coords,_=unpack(mesh,y,gamma,eps); vals=[]
    for ee,uid in enumerate(mesh['eu']):
        ve=coords[uid]
        for q in range(mesh['N'].shape[1]):
            N=mesh['N'][ee,q]; DG=mesh['G'][ee,q]; Rq=mesh['Rq'][ee,q]
            val=N@ve; der=np.einsum('ia,ic->ca',DG,ve); rr=val[0]
            F=np.array([[der[0,0],0,der[0,1]],[rr*der[1,0],rr/Rq,rr*der[1,1]],[der[2,0],0,der[2,1]]])
            vals.append(np.linalg.det(F))
    return float(np.min(vals)),float(np.max(vals))

def fit_profile(r,w,dp,gammaobs,aobs):
    x=r/aobs; A=np.c_[np.ones_like(x),x*x]; sw=np.sqrt(np.maximum(w,0)); C=np.linalg.lstsq(A*sw[:,None],dp*sw,rcond=None)[0]; pred=A@C
    C0,C2=C; q=-(C0+C2)/gammaobs**2/1000.; alpha=(3*C0+C2)/gammaobs**2/1000.
    rmse=math.sqrt(float(np.sum(w*(dp-pred)**2)/np.sum(w))); mu=np.sum(w*dp)/np.sum(w); sst=np.sum(w*(dp-mu)**2); r2=1-np.sum(w*(dp-pred)**2)/sst
    return dict(C0_Pa=float(C0),C2_Pa=float(C2),q_inv_kPa=float(q),alpha_inv_kPa=float(alpha),rmse_Pa=rmse,R2=float(r2),pred=pred)

def run(nr=8,nz=4,edge_cells=4,aspect=5.,eps=.05,gamma=.2,K=10000.,beta_r=1000.,edge_width=.1,grading_power=None,pre_step=.005,gamma_step=.05,beta_theta=None,beta_z=None,plate_mode='tapered_springs'):
    t=time.time(); mesh=build(nr,nz,aspect,edge_width,edge_cells,grading_power,plate_mode); y=initial(mesh); hist=[]; ok=True
    # warm compile local kernels
    _=assemble(mesh,y,K,0.,0.,True,beta_r,beta_theta,beta_z)
    for e in np.linspace(min(pre_step,eps),eps,max(1,int(math.ceil(eps/pre_step)))):
        y,oo,h=newton(mesh,y,K,0.,float(e),beta_r,beta_theta,beta_z); hist+=h; ok &= oo
        if not oo: break
    y0=y.copy(); p0=top_stress(mesh,y0,K,0.,eps)
    if ok:
        for g in np.linspace(min(gamma_step,gamma),gamma,max(1,int(math.ceil(gamma/gamma_step)))):
            y,oo,h=newton(mesh,y,K,float(g),eps,beta_r,beta_theta,beta_z); hist+=h; ok &= oo
            if not oo: break
    pg=top_stress(mesh,y,K,gamma,eps); coords,_=unpack(mesh,y,gamma,eps); top=np.isclose(mesh['X'][:,1],mesh['H']); aobs=float(np.max(coords[top,0])); gammaobs=gamma*aobs/R
    order=np.argsort(p0['r_cur']); base=np.interp(pg['r_cur'],p0['r_cur'][order],p0['szz_Pa'][order]); dp=-(pg['szz_Pa']-base)
    ft=fit_profile(pg['r_cur'],pg['a_cur'],dp,gammaobs,aobs); alpha_ref=G_KPA*(1-eps)**2; Jmin,Jmax=jacobian_range(mesh,y,gamma,eps)
    # contamination projection relative homogeneous zero-q reference
    x=pg['r_cur']/aobs; pref=gammaobs**2*1000*(alpha_ref/2-alpha_ref/2*x*x); contam=dp-pref
    A=np.c_[np.ones_like(x),x*x]; sw=np.sqrt(np.maximum(pg['a_cur'],0)); dc=np.linalg.lstsq(A*sw[:,None],contam*sw,rcond=None)[0]; par=A@dc; perp=contam-par
    norm=lambda z: math.sqrt(float(np.sum(pg['a_cur']*z*z)/np.sum(pg['a_cur']))); scale=alpha_ref*gammaobs**2*1000
    row={k:v for k,v in ft.items() if k!='pred'}
    row.update(dict(nr=nr,nz=nz,edge_cells=mesh['edge_cells'],grading_power=grading_power,aspect=aspect,eps=eps,gamma=gamma,K_over_G=K,beta_r=beta_r,edge_width=edge_width,ok=bool(ok),res_inf=float(hist[-1][0]),iters=len(hist),elapsed_s=time.time()-t,aobs=aobs,gammaobs=gammaobs,alpha_ref_kPa=alpha_ref,rho_false_pct=100*abs(ft['q_inv_kPa'])/alpha_ref,alpha_bias_pct=100*(ft['alpha_inv_kPa']/alpha_ref-1),dC0_Pa=float(dc[0]),dC2_Pa=float(dc[1]),Mpar_pct=100*norm(par)/scale,Mperp_pct=100*norm(perp)/scale,Fpar=float(norm(par)**2/(norm(par)**2+norm(perp)**2)),Jmin=Jmin,Jmax=Jmax,admissible=bool(ok and Jmin>0.2 and np.isfinite(ft['q_inv_kPa'])),formulation='independently assembled sparse Q2/Q1 Taylor-Hood; Eulerian subtraction; consistently tapered plate springs'))
    row['profile']=dict(r_cur=pg['r_cur'].tolist(),a_cur=pg['a_cur'].tolist(),dp_Pa=dp.tolist(),p0_r_cur=p0['r_cur'].tolist(),p0_szz_Pa=p0['szz_Pa'].tolist(),pg_szz_Pa=pg['szz_Pa'].tolist())
    return row

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--nr',type=int,default=8); ap.add_argument('--nz',type=int,default=4); ap.add_argument('--edge-cells',type=int,default=4); ap.add_argument('--aspect',type=float,default=5.); ap.add_argument('--eps',type=float,default=.05); ap.add_argument('--gamma',type=float,default=.2); ap.add_argument('--K',type=float,default=10000.); ap.add_argument('--beta-r',type=float,default=1000.); ap.add_argument('--edge-width',type=float,default=.1); ap.add_argument('--grading-power',type=float,default=None); ap.add_argument('--pre-step',type=float,default=.005); ap.add_argument('--gamma-step',type=float,default=.05); ap.add_argument('--out',required=True); a=ap.parse_args()
    d=run(a.nr,a.nz,a.edge_cells,a.aspect,a.eps,a.gamma,a.K,a.beta_r,a.edge_width,a.grading_power,a.pre_step,a.gamma_step); Path(a.out).write_text(json.dumps(d,indent=2)); print(json.dumps({k:v for k,v in d.items() if k!='profile'},indent=2))
