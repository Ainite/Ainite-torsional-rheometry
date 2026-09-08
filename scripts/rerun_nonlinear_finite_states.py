#!/usr/bin/env python3
"""Reproduce the principal 38x40 nonlinear finite-state comparisons.

The script reconstructs the Mooney-Rivlin baseline energy from the supplied
solver source, solves (i) the finite constitutive null-space perturbation,
(ii) the coordinated interface state with |Delta log beta_i|=0.25, and
(iii) the coordinated interface state with |Delta log beta_i|=0.32.
It then evaluates the common 25-point pressure observation and observed radius.
"""
from __future__ import annotations
from pathlib import Path
import importlib.util
import json
import math
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INP = ROOT / "input"
OUT = ROOT / "nonlinear_rerun_results"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(INP))
import postprocess_core as post

BASE_BETA = np.array([1000.0, 3000.0, 3000.0])
C_PROBE = 0.7644117178
SIGMA_P = 2.0
R0_MM = 30.0

OLD_BLOCK = """    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    Wi=.5*(Js**(-2/3)*I1-3.)\n"""
MR_BLOCK = """    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    C2=C@C; I2=.5*(I1*I1-jnp.trace(C2,axis1=-2,axis2=-1))\n    f=0.30*0.95/(1.0-0.30*(1.0-0.95))\n    I1b=Js**(-2/3)*I1; I2b=Js**(-4/3)*I2\n    Wi=.5*(1-f)*(I1b-3.)+.5*f*(I2b-3.)\n"""
PROBE_BLOCK = """    C=jnp.swapaxes(F,-1,-2)@F; I1=jnp.trace(C,axis1=-2,axis2=-1)\n    C2=C@C; I2=.5*(I1*I1-jnp.trace(C2,axis1=-2,axis2=-1))\n    f=0.30*0.95/(1.0-0.30*(1.0-0.95))\n    I1b=Js**(-2/3)*I1; I2b=Js**(-4/3)*I2\n    lam=0.95\n    I10=lam*lam+2.0/lam; I20=2.0*lam+lam**(-2)\n    h=(I1b-I10)-lam*(I2b-I20)\n    cprobe=0.7644117178\n    Wi=.5*(1-f)*(I1b-3.)+.5*f*(I2b-3.)+cprobe*h*h\n"""


def load_solver(name: str, energy_block: str):
    src = (INP / "solver_core.py").read_text(encoding="utf-8")
    if OLD_BLOCK not in src:
        raise RuntimeError("solver_core.py does not contain the expected baseline energy block")
    tmp = OUT / f"_{name}_runtime.py"
    tmp.write_text(src.replace(OLD_BLOCK, energy_block), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, tmp)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    # Exact 38x40 graded radial mesh used by the supplied baseline state.
    orig = mod.radial_knots
    R = mod.R
    def radial_knots(nr, edge_width=.1, edge_cells=None, grading_power=None):
        if int(nr) == 38 and int(edge_cells or 0) == 16:
            bulk = 22
            r0 = R * (1 - edge_width)
            bulkarr = np.linspace(0.0, r0, bulk + 1)
            u = np.arange(1, 17, dtype=float) / 16.0
            edge = r0 + R * edge_width * (1 - (1 - u) ** 2.0)
            return np.r_[bulkarr, edge]
        return orig(nr, edge_width, edge_cells, grading_power)
    mod.radial_knots = radial_knots
    return mod


def build(mod):
    return mod.build(38, 40, 5.0, 0.1, 16, None, "tapered_springs")


def raw_profile(mod, mesh, yb, ys, cstress: float):
    p0 = post.top_stress(mod, mesh, yb, 0.0, 0.05, post.F_MR, cstress, 0.0, 0.0)
    pg = post.top_stress(mod, mesh, ys, 0.2, 0.05, post.F_MR, cstress, 0.0, 0.0)
    coords, _ = mod.unpack(mesh, ys, 0.2, 0.05)
    top = np.isclose(mesh["X"][:, 1], mesh["H"])
    aobs = float(np.max(coords[top, 0]))
    order = np.argsort(p0["r_cur"])
    basep = np.interp(pg["r_cur"], p0["r_cur"][order], p0["szz_Pa"][order])
    x = np.asarray(pg["r_cur"]) / aobs
    p = -(np.asarray(pg["szz_Pa"]) - basep)
    return x, p, aobs


def equal_area_points():
    j = np.arange(25, dtype=float)
    return np.sqrt(0.05**2 + (j + 0.5) / 25.0 * (0.85**2 - 0.05**2))


def sample(x, y, xs):
    o = np.argsort(x)
    return np.interp(xs, np.asarray(x)[o], np.asarray(y)[o])


def metrics(con, fix, ac, af):
    nc = float(np.linalg.norm(con) / SIGMA_P)
    nf = float(np.linalg.norm(fix) / SIGMA_P)
    residual = float(np.linalg.norm(con - fix) / SIGMA_P)
    angle = float(math.degrees(math.acos(np.clip(np.dot(con, fix) / (np.linalg.norm(con) * np.linalg.norm(fix)), -1.0, 1.0))))
    eta = float(np.dot(con, fix) / np.dot(fix, fix))
    prof = float(np.linalg.norm(con - eta * fix) / SIGMA_P)
    return {
        "constitutive_norm_sigma": nc,
        "interface_norm_sigma": nf,
        "direct_residual_sigma": residual,
        "direction_angle_deg": angle,
        "optimal_scalar": eta,
        "direction_profiled_residual_sigma": prof,
        "constitutive_radius_shift_um": float((ac - ABASE) * R0_MM * 1000.0),
        "interface_radius_shift_um": float((af - ABASE) * R0_MM * 1000.0),
        "radius_mismatch_um": float((af - ac) * R0_MM * 1000.0),
    }


mr = load_solver("solver_mr", MR_BLOCK)
probe = load_solver("solver_probe", PROBE_BLOCK)
mesh_mr = build(mr)
mesh_probe = build(probe)
z = np.load(INP / "baseline_state_38x40.npz")
yb0 = np.asarray(z["yb0"], float)
ys0 = np.asarray(z["ys0"], float)

# Baseline pressure field.
x0, p0, ABASE = raw_profile(mr, mesh_mr, yb0, ys0, 0.0)
xs = equal_area_points()
base = sample(x0, p0, xs)

# Finite constitutive state.
ybc, okbc, hbc = probe.newton(mesh_probe, yb0.copy(), 10000.0, 0.0, 0.05, *BASE_BETA, tol=1e-8, maxit=20)
ysc, oksc, hsc = probe.newton(mesh_probe, ys0.copy(), 10000.0, 0.2, 0.05, *BASE_BETA, tol=1e-8, maxit=20)
if not (okbc and oksc):
    raise RuntimeError("finite constitutive solve failed")
xc, pc, ac = raw_profile(probe, mesh_probe, ybc, ysc, C_PROBE)
con = sample(xc, pc, xs) - base

records = {
    "constitutive": {
        "c": C_PROBE,
        "newton_pre": hbc,
        "newton_twist": hsc,
        "J_pre": list(map(float, probe.jacobian_range(mesh_probe, ybc, 0.0, 0.05))),
        "J_twist": list(map(float, probe.jacobian_range(mesh_probe, ysc, 0.2, 0.05))),
        "radius_nondim": ac,
    }
}
profiles = {"x": xs, "finite_constitutive_shift_Pa": con}
np.savez_compressed(OUT / "finite_constitutive_state_38x40.npz", yb=ybc, ys=ysc, c=C_PROBE)

for bound, tag in [(0.25, "025"), (0.32, "032")]:
    dlog = np.array([-bound, bound, bound])
    bet = BASE_BETA * np.exp(dlog)
    yb, okb, hb = mr.newton(mesh_mr, yb0.copy(), 10000.0, 0.0, 0.05, *bet, tol=1e-8, maxit=20)
    ys, oks, hs = mr.newton(mesh_mr, ys0.copy(), 10000.0, 0.2, 0.05, *bet, tol=1e-8, maxit=20)
    if not (okb and oks):
        raise RuntimeError(f"coordinated interface solve failed for bound={bound}")
    xf, pf, af = raw_profile(mr, mesh_mr, yb, ys, 0.0)
    fix = sample(xf, pf, xs) - base
    rec = metrics(con, fix, ac, af)
    rec.update({
        "bound": bound,
        "delta_log_beta": dlog.tolist(),
        "betas": bet.tolist(),
        "newton_pre": hb,
        "newton_twist": hs,
        "J_pre": list(map(float, mr.jacobian_range(mesh_mr, yb, 0.0, 0.05))),
        "J_twist": list(map(float, mr.jacobian_range(mesh_mr, ys, 0.2, 0.05))),
        "radius_nondim": af,
    })
    records[f"interface_{tag}"] = rec
    profiles[f"interface_{tag}_shift_Pa"] = fix
    profiles[f"constitutive_minus_interface_{tag}_Pa"] = con - fix
    np.savez_compressed(OUT / f"coordinated_interface_{tag}_state_38x40.npz", yb=yb, ys=ys, delta_log_beta=dlog, betas=bet)

# Pressure-fit coefficient increments for direct comparison.
X = np.c_[np.ones_like(xs), xs * xs]
records["finite_constitutive_dAB_Pa"] = np.linalg.lstsq(X, con, rcond=None)[0].tolist()
for tag in ["025", "032"]:
    records[f"interface_{tag}_dAB_Pa"] = np.linalg.lstsq(X, profiles[f"interface_{tag}_shift_Pa"], rcond=None)[0].tolist()
records["baseline_radius_nondim"] = ABASE

# Write results.
cols = list(profiles)
arr = np.column_stack([profiles[k] for k in cols])
np.savetxt(OUT / "finite_pressure_profiles.csv", arr, delimiter=",", header=",".join(cols), comments="")
(OUT / "finite_state_results.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
print(json.dumps(records, indent=2))
