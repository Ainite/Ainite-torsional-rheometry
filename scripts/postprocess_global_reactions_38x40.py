#!/usr/bin/env python3
from pathlib import Path
import importlib.util, sys, json, math
import numpy as np
ROOT=Path(__file__).resolve().parent.parent; INP=ROOT/'input'; OUT=ROOT/'results'; sys.path.insert(0,str(INP)); import postprocess_core as post
OLD="""    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    Wi=.5*(Js**(-2/3)*I1-3.)\n"""
MR="""    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    C2=C@C; I2=.5*(I1*I1-jnp.trace(C2,axis1=-2,axis2=-1))\n    f=0.30*0.95/(1.0-0.30*(1.0-0.95))\n    I1b=Js**(-2/3)*I1; I2b=Js**(-4/3)*I2\n    Wi=.5*(1-f)*(I1b-3.)+.5*f*(I2b-3.)\n"""
PROBE="""    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    C2=C@C; I2=.5*(I1*I1-jnp.trace(C2,axis1=-2,axis2=-1))\n    f=0.30*0.95/(1.0-0.30*(1.0-0.95))\n    I1b=Js**(-2/3)*I1; I2b=Js**(-4/3)*I2\n    lam=0.95\n    I10=lam*lam+2.0/lam; I20=2.0*lam+lam**(-2)\n    h=(I1b-I10)-lam*(I2b-I20)\n    cprobe=0.7644117178\n    Wi=.5*(1-f)*(I1b-3.)+.5*f*(I2b-3.)+cprobe*h*h\n"""
def load(name,block):
 src=(INP/'solver_core.py').read_text(); tmp=Path('/mnt/data')/f'_{name}.py'; tmp.write_text(src.replace(OLD,block)); sp=importlib.util.spec_from_file_location(name,tmp); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
 orig=m.radial_knots; R=m.R
 def rk(nr,edge_width=.1,edge_cells=None,grading_power=None):
  if int(nr)==38 and int(edge_cells or 0)==16:
   bulk=22; r0=R*(1-edge_width); bulkarr=np.linspace(0.,r0,bulk+1); u=np.arange(1,17,dtype=float)/16.; edge=r0+R*edge_width*(1-(1-u)**2); return np.r_[bulkarr,edge]
  return orig(nr,edge_width,edge_cells,grading_power)
 m.radial_knots=rk; return m

def reactions(sq,mesh,y,gamma,eps,c):
 coords,p=sq.unpack(mesh,y,gamma,eps); GPa=float(sq.G_KPA)*1000.; R0=0.03
 Fz=0.; T=0.
 for i in range(mesh['nr']):
  ei=(mesh['nz']-1)*mesh['nr']+i; uid=mesh['eu'][ei]; pid=mesh['ep'][ei]; Xe=mesh['X'][uid]; ve=coords[uid]; pe=p[pid]
  eta=1.; le,dle=sq.L2(eta); lp_e,_=sq.L1(eta)
  for ia,xi in enumerate(sq.GP3):
   lx,dlx=sq.L2(xi); lp_x,_=sq.L1(xi); N=[]; dxi=[]; deta=[]
   for bb in range(3):
    for aa in range(3): N.append(lx[aa]*le[bb]); dxi.append(dlx[aa]*le[bb]); deta.append(lx[aa]*dle[bb])
   N=np.asarray(N); dxi=np.asarray(dxi); deta=np.asarray(deta); J2=np.array([[dxi@Xe[:,0],deta@Xe[:,0]],[dxi@Xe[:,1],deta@Xe[:,1]]]); DG=np.column_stack([dxi,deta])@np.linalg.inv(J2)
   val=N@ve; der=np.einsum('ia,ic->ca',DG,ve); rr=float(val[0]); Rq=float(N@Xe[:,0])
   F=np.array([[der[0,0],0.,der[0,1]],[rr*der[1,0],rr/Rq,rr*der[1,1]],[der[2,0],0.,der[2,1]]]); Npv=np.array([lp_x[0]*lp_e[0],lp_x[1]*lp_e[0],lp_x[1]*lp_e[1],lp_x[0]*lp_e[1]]); pq=float(Npv@pe)
   P=post.piso(F,post.F_MR,c,0.,0.)+pq*np.linalg.inv(F).T
   dR=float(dxi@Xe[:,0]); dA0=2*math.pi*Rq*dR*float(sq.GW3[ia])
   Fz += float(P[2,2])*GPa*dA0*R0**2
   T += float(P[1,2])*GPa*rr*dA0*R0**3
 return {'axial_force_N':Fz,'torque_Nm':T}
mr=load('rxmr',MR); probe=load('rxprobe',PROBE); mm=mr.build(38,40,5.,.1,16,None,'tapered_springs'); mp=probe.build(38,40,5.,.1,16,None,'tapered_springs')
z=np.load(INP/'baseline_state_38x40.npz'); zb={'yb':z['yb0'],'ys':z['ys0']}; zc=np.load(INP/'finite_constitutive_state_38x40.npz'); zi=np.load(INP/'coordinated_interface_032_state_38x40.npz')
out={}
for name,sq,mesh,st,c in [('baseline',mr,mm,zb,0.),('constitutive',probe,mp,zc,0.7644117178),('interface_032',mr,mm,zi,0.)]:
 out[name]={'pre':reactions(sq,mesh,st['yb'],0.,.05,c),'twist':reactions(sq,mesh,st['ys'],.2,.05,c)}
for key in ['axial_force_N','torque_Nm']:
 a=out['constitutive']['twist'][key]; b=out['interface_032']['twist'][key]; out.setdefault('constitutive_vs_interface_032',{})[key+'_difference_interface_minus_constitutive']=b-a; out['constitutive_vs_interface_032'][key+'_relative_to_constitutive_pct']=(b-a)/a*100 if a else None
 # twist-induced increment relative to own pre state
 ac=out['constitutive']['twist'][key]-out['constitutive']['pre'][key]; ai=out['interface_032']['twist'][key]-out['interface_032']['pre'][key]; out['constitutive_vs_interface_032'][key+'_twist_increment_constitutive']=ac; out['constitutive_vs_interface_032'][key+'_twist_increment_interface']=ai; out['constitutive_vs_interface_032'][key+'_twist_increment_difference']=ai-ac; out['constitutive_vs_interface_032'][key+'_twist_increment_relative_pct']=(ai-ac)/ac*100 if ac else None
(OUT/'global_reactions_38x40.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
