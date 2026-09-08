#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import importlib.util, json, math, sys, time, os
import numpy as np
import scipy.sparse.linalg as spla
import jax, jax.numpy as jnp

HERE=Path(__file__).resolve().parent; INP=HERE/'input'; OUT=HERE/'results'; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(INP)); import postprocess_core as post
K=10000.; EPS=.05; GAM=.2; BETA=np.array([1000.,3000.,3000.]); NR=34; NZ=20; EDGE_CELLS=10; EDGE_WIDTH=.1
OLD="""    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    Wi=.5*(Js**(-2/3)*I1-3.)\n"""
MR="""    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    C2=C@C; I2=.5*(I1*I1-jnp.trace(C2,axis1=-2,axis2=-1))\n    f=0.30*0.95/(1.0-0.30*(1.0-0.95))\n    I1b=Js**(-2/3)*I1; I2b=Js**(-4/3)*I2\n    Wi=.5*(1-f)*(I1b-3.)+.5*f*(I2b-3.)\n"""

def load_mr():
    src=(INP/'solver_core.py').read_text(); tmp=OUT/'_r181_mr.py'; tmp.write_text(src.replace(OLD,MR))
    sp=importlib.util.spec_from_file_location('r181_mr',tmp); m=importlib.util.module_from_spec(sp); assert sp.loader; sp.loader.exec_module(m); return m
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

def smooth_inner_weight(r,center,halfwidth):
    r=np.asarray(r,float)/float(m.R); c=float(center); d=float(halfwidth)
    lo,hi=c-d,c+d; t=np.clip((r-lo)/(hi-lo),0.,1.); s=6*t**5-15*t**4+10*t**3; w=1-s
    return np.where(r<=lo,1.,np.where(r>=hi,0.,w))

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
        target=np.array([rr, theta if is_top else 0., H*(1-EPS) if is_top else 0.]); delta=cur[c]-target[c]
        scales=np.array([1.,rr,1.]); kk=kbase[c]*wt*tap*scales[c]**2
        smap=smaps[c]; active=np.flatnonzero(smap>=0); gm=smap[active]; Na=Ns[active]
        np.add.at(gg,gm,kk*delta*Na)  # derivative wrt log beta_c
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
            w=1.0 if center is None else float(smooth_inner_weight(np.array([Rq]),center,halfwidth)[0])
            P=post.piso(F,post.F_MR,c*w,0.,0.)+pq*np.linalg.inv(F).T; sig=(P@F.T)/J*G0*1000.; out['r_cur'].append(rr); out['szz_Pa'].append(sig[2,2])
    return {k:np.asarray(v) for k,v in out.items()}

def raw_prof(mesh,ybp,ysp,c=0.,center=None,halfwidth=.02):
    p0=top_stress_weighted(mesh,ybp,0.,c,center,halfwidth); pg=top_stress_weighted(mesh,ysp,GAM,c,center,halfwidth); coords,_=m.unpack(mesh,ysp,GAM,EPS); top=np.isclose(mesh['X'][:,1],mesh['H']); a=float(np.max(coords[top,0])); o=np.argsort(p0['r_cur']); bp=np.interp(pg['r_cur'],p0['r_cur'][o],p0['szz_Pa'][o]); return pg['r_cur']/a,-(pg['szz_Pa']-bp),a

def tangent_profile(mesh,yb,ys,dyb,dys,direct_c=0.,center=None,halfwidth=.02):
    vals={}
    for h in [1e-4,5e-5]:
        xp,pp,ap=raw_prof(mesh,yb+h*dyb,ys+h*dys,+h*direct_c,center,halfwidth); xm,pm,am=raw_prof(mesh,yb-h*dyb,ys-h*dys,-h*direct_c,center,halfwidth)
        vals[h]=((sample(xp,pp)-sample(xm,pm))/(2*h),(ap-am)/(2*h))
    return vals[5e-5][0], angle(vals[1e-4][0],vals[5e-5][0]), float(vals[5e-5][1])

def continuation(mesh):
    y=m.initial(mesh); hist=[]
    # warm compilation
    _=m.assemble(mesh,y,K,0.,0.,True,*BETA)
    for e in np.arange(.005,.0501,.005):
        y,ok,h=m.newton(mesh,y,K,0.,float(e),*BETA,tol=1e-8,maxit=40); hist+=h
        if not ok: raise RuntimeError(f'pre failed {e} {h[-1]}')
    yb=y.copy()
    for g in [.05,.10,.15,.20]:
        y,ok,h=m.newton(mesh,y,K,g,EPS,*BETA,tol=1e-8,maxit=40); hist+=h
        if not ok: raise RuntimeError(f'twist failed {g} {h[-1]}')
    return yb,y.copy(),hist

def project(g,U):
    r=g-U@(U.T@g); n=np.linalg.norm(g); return {'norm':float(n),'orth_norm':float(np.linalg.norm(r)),'orth_fraction':float(np.linalg.norm(r)/n),'angle_deg':float(math.degrees(math.asin(np.clip(np.linalg.norm(r)/n,0,1))))}

def run_case(tag,aspect):
    t0=time.time(); mesh=m.build(NR,NZ,aspect,EDGE_WIDTH,EDGE_CELLS,None,'tapered_springs')
    state_path=OUT/f'baseline_{tag}.npz'
    if state_path.exists():
        zz=np.load(state_path); yb=np.asarray(zz['yb'],float); ys=np.asarray(zz['ys'],float); hist=[[float('nan'),float('nan')]]
    elif aspect==2.5 and (OUT/'baseline_aspect5_independent_radial.npz').exists():
        # Same 34x20 topology. Scale the converged aspect-5 current coordinates to H/R=0.4 as a geometry-consistent Newton seed.
        z5=np.load(OUT/'baseline_aspect5_independent_radial.npz'); mesh5=m.build(NR,NZ,5.0,EDGE_WIDTH,EDGE_CELLS,None,'tapered_springs')
        def scale_state(y5,gamma):
            c5,p5=m.unpack(mesh5,np.asarray(y5,float),gamma,EPS); c2=c5.copy(); c2[:,1]*=2.0; c2[:,2]*=2.0; flat=c2.reshape(-1); return np.r_[flat[mesh['free_flat']],p5]
        yb0=scale_state(z5['yb'],0.0); ys0=scale_state(z5['ys'],GAM)
        yb,okb,hb=m.newton(mesh,yb0,K,0.,EPS,*BETA,tol=1e-8,maxit=50); ys,oks,hs=m.newton(mesh,ys0,K,GAM,EPS,*BETA,tol=1e-8,maxit=50); hist=hb+hs
        if not(okb and oks): raise RuntimeError(f'scaled geometry seed failed: pre={okb}, twist={oks}, last={hist[-1]}')
    else:
        yb,ys,hist=continuation(mesh)
    np.savez_compressed(state_path,yb=yb,ys=ys,rp=mesh['rp'],H=mesh['H'])
    Eb,gb,Hb=m.assemble(mesh,yb,K,0.,EPS,True,*BETA); Es,gs,Hs=m.assemble(mesh,ys,K,GAM,EPS,True,*BETA)
    # interface tangent columns, each recomputed for this mesh/geometry
    cols=[]; beta_step=[]
    for c in range(3):
        sb=beta_residual_derivative(mesh,yb,0.,c); ss=beta_residual_derivative(mesh,ys,GAM,c); dyb=spla.spsolve(Hb,-sb); dys=spla.spsolve(Hs,-ss)
        g,sa,rd=tangent_profile(mesh,yb,ys,dyb,dys,0.,None,.02); cols.append(g); beta_step.append({'channel':c,'step_angle_deg':sa,'radius_deriv':rd})
    B=np.column_stack(cols); U,sv,_=np.linalg.svd(B,full_matrices=False); orth=lambda g:g-U@(U.T@g)
    # full source tangent
    mask=np.ones_like(np.asarray(mesh['Rq'])); sb=assemble_source(mesh,yb,mask,True); ss=assemble_source(mesh,ys,mask,False); dyb=spla.spsolve(Hb,-sb); dys=spla.spsolve(Hs,-ss); full,full_step,full_rad=tangent_profile(mesh,yb,ys,dyb,dys,1.,None,.02)
    # smooth cumulative windows
    rows=[]; prof={}
    for center in [.80,.82,.84,.86,.88,.90]:
        mask=smooth_inner_weight(np.asarray(mesh['Rq']),center,.02); sb=assemble_source(mesh,yb,mask,True); ss=assemble_source(mesh,ys,mask,False); dyb=spla.spsolve(Hb,-sb); dys=spla.spsolve(Hs,-ss); g,sa,rd=tangent_profile(mesh,yb,ys,dyb,dys,1.,center,.02); prof[center]=g; rows.append({'center':center,'halfwidth':.02,**project(g,U),'step_angle_deg':sa,'radius_deriv':rd,'mask_mass_fraction':float(np.sum(mask*np.asarray(mesh['W']))/np.sum(np.asarray(mesh['W'])))})
    a=prof[.82]; ex=prof[.88]; band=ex-a; oa=orth(a); ob=orth(band); os=orth(ex); ang=angle(oa,ob); red=1-float(np.linalg.norm(os)/np.linalg.norm(oa)); pass_cancel=bool(ang>150 and red>=.50)
    fr=np.array([r['orth_fraction'] for r in rows]); drops=fr[:-1]-fr[1:]; imax=int(np.argmax(drops)); collapse_interval=[rows[imax]['center'],rows[imax+1]['center']]; collapse_mid=.5*sum(collapse_interval); pass_location=bool(.85<=collapse_mid<=.89)
    valid=bool(max([full_step]+[x['step_angle_deg'] for x in beta_step]+[r['step_angle_deg'] for r in rows])<.01 and np.linalg.norm(gb,np.inf)<1e-7 and np.linalg.norm(gs,np.inf)<1e-7)
    rec={'tag':tag,'aspect':aspect,'H0_over_R0':1/aspect,'mesh':f'{NR}x{NZ}_ec{EDGE_CELLS}','bulk_cells':NR-EDGE_CELLS,'radial_knots':mesh['rp'].tolist(),'baseline_grad_inf_pre':float(np.linalg.norm(gb,np.inf)),'baseline_grad_inf_twist':float(np.linalg.norm(gs,np.inf)),'newton_terminal_grad_inf':None if not np.isfinite(hist[-1][0]) else float(hist[-1][0]),'interface_singular_values_Pa_per_logbeta':sv.tolist(),'interface_step_checks':beta_step,'full':{**project(full,U),'step_angle_deg':full_step,'radius_deriv':full_rad},'window_rows':rows,'cancellation':{'interior_center':.82,'expanded_center':.88,'halfwidth':.02,'orth_angle_deg':ang,'orth_norm_before':float(np.linalg.norm(oa)),'orth_norm_added_band':float(np.linalg.norm(ob)),'orth_norm_after':float(np.linalg.norm(os)),'reduction_fraction':red,'pass':pass_cancel},'collapse_location':{'largest_fraction_drop':float(drops[imax]),'interval':collapse_interval,'midpoint':collapse_mid,'pass_preinterface_location':pass_location},'valid':valid,'runtime_s':time.time()-t0}
    np.savetxt(OUT/f'profiles_{tag}.csv',np.column_stack([XS,cols[0],cols[1],cols[2],full]+[prof[c] for c in [.80,.82,.84,.86,.88,.90]]),delimiter=',',header='x,beta_r_log,beta_theta_log,beta_z_log,full,c080,c082,c084,c086,c088,c090',comments='')
    print(json.dumps({'tag':tag,'valid':valid,'cancellation':rec['cancellation'],'collapse':rec['collapse_location'],'full_orth_fraction':rec['full']['orth_fraction'],'runtime_s':rec['runtime_s']},indent=2),flush=True)
    return rec

rule={'reference':'R180 38x40 smooth-support gate; feasibility amendment reduces only axial count from 40 to 20 before any R181 scientific output','new_discretization':{'nr':NR,'nz':NZ,'edge_cells':EDGE_CELLS,'bulk_cells':NR-EDGE_CELLS,'note':'bulk radial partition differs from R180: 24 rather than 22 cells on 0<=R/R0<=0.9; axial count reduced to 20 by pre-result feasibility amendment'},'cases':[{'tag':'aspect5_independent_radial','H0_over_R0':.2},{'tag':'aspect2p5_independent_geometry','H0_over_R0':.4}],'smooth_window':{'halfwidth':.02,'centers':[.80,.82,.84,.86,.88,.90],'cancellation_pair':[.82,.88]},'replication_gate':'valid numerical gate AND orthogonal-component angle >150 deg AND orthogonal norm reduction >=50%','location_gate':'largest consecutive orthogonal-fraction drop midpoint in [0.85,0.89], i.e. immediately inside physical transition onset R/R0=0.90','validity':'all centered step-angle checks <0.01 deg and baseline residual infinity norms <1e-7'}
(OUT/'R181_PREREGISTERED_RULE.json').write_text(json.dumps(rule,indent=2)+'\n')
print('PREREGISTERED',json.dumps(rule,indent=2),flush=True)
sel=os.environ.get('R181_CASE_ONLY','').strip()
if sel:
    mapping={'aspect5_independent_radial':5.0,'aspect2p5_independent_geometry':2.5}
    if sel not in mapping: raise SystemExit(f'unknown R181_CASE_ONLY={sel}')
    one=run_case(sel,mapping[sel]); (OUT/f'case_{sel}.json').write_text(json.dumps(one,indent=2)+'\n'); raise SystemExit(0)
cases=[run_case('aspect5_independent_radial',5.0),run_case('aspect2p5_independent_geometry',2.5)]
A,B=cases
if not all(c['valid'] for c in cases): verdict='INVALID_NUMERICAL_GATE'
elif A['cancellation']['pass'] and B['cancellation']['pass'] and A['collapse_location']['pass_preinterface_location'] and B['collapse_location']['pass_preinterface_location']:
    verdict='COMBINED_CANCELLATION_AND_LOCATION_REPLICATED'
elif A['cancellation']['pass'] and B['cancellation']['pass'] and (not A['collapse_location']['pass_preinterface_location']) and (not B['collapse_location']['pass_preinterface_location']):
    verdict='CANCELLATION_REPLICATED_LOCATION_NOT_REPLICATED'
elif A['cancellation']['pass'] and B['cancellation']['pass']:
    verdict='CANCELLATION_REPLICATED_LOCATION_MIXED'
else:
    verdict='CANCELLATION_NOT_REPLICATED'
out={'schema':'U1-R181-cross-replication-v1','preregistered_rule':rule,'cases':cases,'verdict':verdict}
(OUT/'R181_CROSS_REPLICATION_GATE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'verdict':verdict,'cases':[{'tag':c['tag'],'cancellation':c['cancellation'],'collapse':c['collapse_location'],'full':c['full']} for c in cases]},indent=2),flush=True)
