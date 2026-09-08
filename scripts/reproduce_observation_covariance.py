#!/usr/bin/env python3
"""Recompute the primary 25-point pressure-fit covariance and derived scales."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
N=25; xmin=0.05; xmax=0.85; sigma_p=2.0; gamma=0.2
j=np.arange(N,dtype=float)
x=np.sqrt(xmin*xmin+(j+0.5)/N*(xmax*xmax-xmin*xmin))
X=np.column_stack([np.ones(N),x*x])
C=sigma_p**2*np.linalg.inv(X.T@X)
wa=np.array([3.0,1.0])/(gamma**2*1000.0)
wq=np.array([-1.0,-1.0])/(gamma**2*1000.0)
out={
  "x":x.tolist(),
  "C_Pa2":C.tolist(),
  "sigma_A_Pa":float(np.sqrt(C[0,0])),
  "sigma_B_Pa":float(np.sqrt(C[1,1])),
  "corr_AB":float(C[0,1]/np.sqrt(C[0,0]*C[1,1])),
  "sigma_AplusB_Pa":float(np.sqrt(np.array([1.,1.])@C@np.array([1.,1.]))),
  "sigma_qP_kPa":float(np.sqrt(wq@C@wq)),
  "sigma_alpha1P_kPa":float(np.sqrt(wa@C@wa)),
}
(OUT/'observation_covariance.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
