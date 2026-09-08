#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, math, time, os
import numpy as np
import scipy.sparse.linalg as spla
import r182_core as C

HERE=Path(__file__).resolve().parent; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
# Frozen before calculation.
RULE={
 'schema':'U1-R182-transition-onset-translation-preregister-v1',
 'mesh':'34x20, aspect=5, edge_cells=10, tapered_springs',
 'edge_widths':[0.05,0.15],
 'transition_onset':'r0=1-edge_width',
 'smooth_halfwidth':0.02,
 'relative_centers_from_onset':[-0.10,-0.08,-0.06,-0.04,-0.02,0.0],
 'cancellation_pair_relative_to_onset':[-0.08,-0.02],
 'causal_support_gate':'for BOTH altered transition widths: valid tangent numerics, orthogonal-component angle >150 deg, and orthogonal norm reduction >=50% from r0-0.08 to r0-0.02',
 'approach_gate':'for BOTH widths, rank-3 orthogonal fraction at r0-0.02 is at least 0.20 lower than at r0-0.08',
 'validity':'all centered step-angle checks <0.01 deg and equilibrium residual infinity norms <1e-7',
 'claim_boundary':'A pass supports translation with physical transition onset for the tested two widths on one amended 34x20 geometry; it does not establish a continuum theorem or width universality.'
}
(OUT/'R182_PREREGISTERED_RULE.json').write_text(json.dumps(RULE,indent=2)+'\n')
print('PREREGISTERED',json.dumps(RULE,indent=2),flush=True)

def proj(g,U):
    r=g-U@(U.T@g); n=np.linalg.norm(g); return {'norm':float(n),'orth_norm':float(np.linalg.norm(r)),'orth_fraction':float(np.linalg.norm(r)/n),'angle_deg':float(math.degrees(math.asin(np.clip(np.linalg.norm(r)/n,0,1))))}

def run_width(w):
    t0=time.time(); r0=1.0-w; mesh=C.m.build(C.NR,C.NZ,5.0,w,C.EDGE_CELLS,None,'tapered_springs')
    state_path=OUT/f'baseline_edgewidth_{w:.2f}.npz'
    if state_path.exists():
        zz=np.load(state_path); yb=np.asarray(zz['yb'],float); ys=np.asarray(zz['ys'],float); hist=[[float('nan'),float('nan')]]
    elif (HERE/'input'/'baseline_edgewidth_0.10.npz').exists():
        # Pre-existing R181 equilibrium state on the same 34x20 topology, used only as a Newton seed.
        zr=np.load(HERE/'input'/'baseline_edgewidth_0.10.npz'); mref=C.m.build(C.NR,C.NZ,5.0,.10,C.EDGE_CELLS,None,'tapered_springs')
        def interp_state(yref,gamma):
            cref,pref=C.m.unpack(mref,np.asarray(yref,float),gamma,C.EPS)
            nr2=2*C.NR+1; nz2=2*C.NZ+1; cr=cref.reshape(nz2,nr2,3); rr_old=np.unique(mref['X'][:,0]); rr_new=np.unique(mesh['X'][:,0]); cn=np.empty((nz2,nr2,3))
            for j in range(nz2):
                for q in range(3): cn[j,:,q]=np.interp(rr_new,rr_old,cr[j,:,q])
            pr=pref.reshape(C.NZ+1,C.NR+1); pn=np.empty_like(pr)
            for j in range(C.NZ+1): pn[j]=np.interp(mesh['rp'],mref['rp'],pr[j])
            flat=cn.reshape(-1,3).reshape(-1); return np.r_[flat[mesh['free_flat']],pn.ravel()]
        yb0=interp_state(zr['yb'],0.0); ys0=interp_state(zr['ys'],C.GAM)
        yb,okb,hb=C.m.newton(mesh,yb0,C.K,0.,C.EPS,*C.BETA,tol=1e-8,maxit=60); ys,oks,hs=C.m.newton(mesh,ys0,C.K,C.GAM,C.EPS,*C.BETA,tol=1e-8,maxit=60); hist=hb+hs
        if not(okb and oks): raise RuntimeError(f'interpolated reference seed failed w={w}: pre={okb}, twist={oks}, last={hist[-1]}')
    else:
        yb,ys,hist=C.continuation(mesh)
    np.savez_compressed(state_path,yb=yb,ys=ys,rp=mesh['rp'],H=mesh['H'],edge_width=w)
    Eb,gb,Hb=C.m.assemble(mesh,yb,C.K,0.,C.EPS,True,*C.BETA); Es,gs,Hs=C.m.assemble(mesh,ys,C.K,C.GAM,C.EPS,True,*C.BETA)
    cols=[]; checks=[]
    for ch in range(3):
        sb=C.beta_residual_derivative(mesh,yb,0.,ch); ss=C.beta_residual_derivative(mesh,ys,C.GAM,ch); dyb=spla.spsolve(Hb,-sb); dys=spla.spsolve(Hs,-ss)
        g,sa,rd=C.tangent_profile(mesh,yb,ys,dyb,dys,0.,None,.02); cols.append(g); checks.append({'channel':ch,'step_angle_deg':sa,'radius_deriv':rd})
    B=np.column_stack(cols); U,sv,_=np.linalg.svd(B,full_matrices=False); orth=lambda g:g-U@(U.T@g)
    # Full source tangent for context.
    ones=np.ones_like(np.asarray(mesh['Rq'])); sb=C.assemble_source(mesh,yb,ones,True); ss=C.assemble_source(mesh,ys,ones,False); dyb=spla.spsolve(Hb,-sb); dys=spla.spsolve(Hs,-ss); full,fsa,frd=C.tangent_profile(mesh,yb,ys,dyb,dys,1.,None,.02)
    rows=[]; profiles={}
    centers=[r0+d for d in RULE['relative_centers_from_onset']]
    for rel,center in zip(RULE['relative_centers_from_onset'],centers):
        mask=C.smooth_inner_weight(np.asarray(mesh['Rq']),center,.02); sb=C.assemble_source(mesh,yb,mask,True); ss=C.assemble_source(mesh,ys,mask,False); dyb=spla.spsolve(Hb,-sb); dys=spla.spsolve(Hs,-ss); g,sa,rd=C.tangent_profile(mesh,yb,ys,dyb,dys,1.,center,.02); profiles[rel]=g; rows.append({'relative_center':rel,'center':center,**proj(g,U),'step_angle_deg':sa,'radius_deriv':rd,'mask_mass_fraction':float(np.sum(mask*np.asarray(mesh['W']))/np.sum(np.asarray(mesh['W'])))})
    a=profiles[-.08]; ex=profiles[-.02]; band=ex-a; oa=orth(a); ob=orth(band); os=orth(ex); ang=C.angle(oa,ob); red=1-float(np.linalg.norm(os)/np.linalg.norm(oa)); cancel_pass=bool(ang>150 and red>=.5)
    f0=next(r['orth_fraction'] for r in rows if abs(r['relative_center']+.08)<1e-12); f1=next(r['orth_fraction'] for r in rows if abs(r['relative_center']+.02)<1e-12); approach_drop=float(f0-f1); approach_pass=bool(approach_drop>=.20)
    maxstep=max([fsa]+[q['step_angle_deg'] for q in checks]+[q['step_angle_deg'] for q in rows]); valid=bool(maxstep<.01 and np.linalg.norm(gb,np.inf)<1e-7 and np.linalg.norm(gs,np.inf)<1e-7)
    rec={'edge_width':w,'transition_onset':r0,'mesh':f'{C.NR}x{C.NZ}_ec{C.EDGE_CELLS}','radial_knots':mesh['rp'].tolist(),'baseline_grad_inf_pre':float(np.linalg.norm(gb,np.inf)),'baseline_grad_inf_twist':float(np.linalg.norm(gs,np.inf)),'interface_singular_values':sv.tolist(),'interface_step_checks':checks,'full':{**proj(full,U),'step_angle_deg':fsa,'radius_deriv':frd},'window_rows':rows,'cancellation':{'pair_relative':[-.08,-.02],'pair_absolute':[r0-.08,r0-.02],'orth_angle_deg':ang,'orth_norm_before':float(np.linalg.norm(oa)),'orth_norm_added_band':float(np.linalg.norm(ob)),'orth_norm_after':float(np.linalg.norm(os)),'reduction_fraction':red,'pass':cancel_pass},'approach':{'orth_fraction_at_minus_0p08':f0,'orth_fraction_at_minus_0p02':f1,'drop':approach_drop,'pass':approach_pass},'max_step_angle_deg':maxstep,'valid':valid,'runtime_s':time.time()-t0}
    np.savetxt(OUT/f'profiles_edgewidth_{w:.2f}.csv',np.column_stack([C.XS,cols[0],cols[1],cols[2],full]+[profiles[d] for d in RULE['relative_centers_from_onset']]),delimiter=',',header='x,beta_r_log,beta_theta_log,beta_z_log,full,'+','.join(f'rel_{d:+.2f}' for d in RULE['relative_centers_from_onset']),comments='')
    print(json.dumps({'edge_width':w,'valid':valid,'cancellation':rec['cancellation'],'approach':rec['approach'],'full_orth_fraction':rec['full']['orth_fraction'],'runtime_s':rec['runtime_s']},indent=2),flush=True)
    return rec

sel=os.environ.get('R182_CASE_ONLY','').strip()
if sel:
    w=float(sel); one=run_width(w); (OUT/f'case_edgewidth_{w:.2f}.json').write_text(json.dumps(one,indent=2)+'\n'); raise SystemExit(0)
cases=[run_width(.05),run_width(.15)]
if not all(c['valid'] for c in cases): verdict='INVALID_NUMERICAL_GATE'
elif all(c['cancellation']['pass'] and c['approach']['pass'] for c in cases): verdict='TRANSITION_ONSET_TRANSLATION_SUPPORTED'
elif all(c['cancellation']['pass'] for c in cases): verdict='RELATIVE_CANCELLATION_REPLICATED_APPROACH_SHAPE_MIXED'
else: verdict='TRANSITION_ONSET_TRANSLATION_NOT_SUPPORTED'
out={'schema':'U1-R182-transition-onset-translation-v1','preregistered_rule':RULE,'cases':cases,'verdict':verdict}
(OUT/'R182_TRANSITION_TRANSLATION_GATE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'verdict':verdict,'cases':[{'edge_width':c['edge_width'],'cancellation':c['cancellation'],'approach':c['approach']} for c in cases]},indent=2),flush=True)
