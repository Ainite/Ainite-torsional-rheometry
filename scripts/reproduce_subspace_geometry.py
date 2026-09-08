#!/usr/bin/env python3
"""Recompute the interface-subspace / quadratic-profile geometry."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INP = ROOT / "input"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Interface tangent columns and scaled tangent constitutive response.
t = np.genfromtxt(INP / "interface_tangent_profiles.csv", delimiter=",", names=True)
x = np.asarray(t["x"], float)
F = np.column_stack([t["beta_r_log"], t["beta_theta_log"], t["beta_z_log"]]).astype(float)
# Finite constitutive profile from the nonlinear comparison.
fp = OUT / "finite_pressure_profiles.csv"
if fp.exists():
    d = np.genfromtxt(fp, delimiter=",", names=True)
    finite = np.asarray(d["finite_constitutive_shift_Pa"], float)
else:
    d = np.genfromtxt(INP / "finite_profile_reference.csv", delimiter=",", names=True)
    finite = np.asarray(d["finite_constitutive_shift_Pa"], float)

# SVD of the physical interface family.
U, s, _ = np.linalg.svd(F, full_matrices=False)
U2 = U[:, :2]
# Quadratic pressure plane span{1,x^2}.
Qx, _ = np.linalg.qr(np.column_stack([np.ones_like(x), x*x]))
# Principal angles via singular values of Qx^T U2.
sv = np.linalg.svd(Qx.T @ U2, compute_uv=False)
sv = np.clip(sv, -1.0, 1.0)
angles = np.degrees(np.arccos(sv))
angles.sort()

sigma = 2.0
def outside(v, Q):
    return float(np.linalg.norm(v - Q @ (Q.T @ v)) / sigma)

res = {
    "principal_singular_values": sv.tolist(),
    "principal_angles_deg": angles.tolist(),
    "finite_constitutive_norm_SD": float(np.linalg.norm(finite)/sigma),
    "outside_quadratic_SD": outside(finite, Qx),
    "outside_rank2_SD": outside(finite, U2),
    "outside_full_span_SD": outside(finite, U),
    "fixture_singular_values_Pa": s.tolist(),
    "fixture_s2_over_s1": float(s[1]/s[0]),
    "fixture_s3_over_s1": float(s[2]/s[0]),
}
for k in ["outside_quadratic_SD","outside_rank2_SD","outside_full_span_SD"]:
    res["fraction_"+k.removeprefix("outside_").removesuffix("_SD")] = res[k]/res["finite_constitutive_norm_SD"]
(OUT / "subspace_geometry.json").write_text(json.dumps(res, indent=2)+"\n", encoding="utf-8")
print(json.dumps(res, indent=2))
