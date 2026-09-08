"""Finite-element top-surface stress post-processing used by the reproduction scripts."""
from __future__ import annotations
import math
import numpy as np

LAM = 0.95
RHO = 0.30
F_MR = RHO * LAM / (1.0 - RHO * (1.0 - LAM))
I10 = LAM * LAM + 2.0 / LAM
I20 = 2.0 * LAM + LAM ** -2


def piso(F, f, c, a=0.0, b=0.0):
    """Pressure-free first Piola stress for the Mooney-Rivlin plus null-space perturbation."""
    F = np.asarray(F)
    J = np.linalg.det(F)
    C = F.T @ F
    I1 = np.trace(C)
    I2 = 0.5 * (I1 * I1 - np.trace(C @ C))
    FinvT = np.linalg.inv(F).T
    I1b = J ** (-2.0 / 3.0) * I1
    I2b = J ** (-4.0 / 3.0) * I2
    Q1 = J ** (-2.0 / 3.0) * (F - (I1 / 3.0) * FinvT)
    D2 = J ** (-4.0 / 3.0) * (
        2.0 * (I1 * F - F @ C) - (4.0 / 3.0) * I2 * FinvT
    )
    u = I1b - I10
    v = I2b - I20
    h = u - LAM * v
    Q = 1.0 + a * u + b * v
    r1 = c * (2.0 * h * Q + h * h * a)
    r2 = c * (-2.0 * LAM * h * Q + h * h * b)
    return (1.0 - f) * Q1 + 0.5 * f * D2 + 2.0 * r1 * Q1 + r2 * D2


def top_stress(sq, mesh, y, gamma, eps, f, c, a=0.0, b=0.0):
    """Evaluate top-face axial Cauchy stress at the radial Gauss points."""
    coords, p = sq.unpack(mesh, y, gamma, eps)
    out = {k: [] for k in ["r_cur", "a_cur", "szz_Pa"]}
    G0 = float(sq.G_KPA)
    for i in range(mesh["nr"]):
        ei = (mesh["nz"] - 1) * mesh["nr"] + i
        uid = mesh["eu"][ei]
        pid = mesh["ep"][ei]
        Xe = mesh["X"][uid]
        ve = coords[uid]
        pe = p[pid]
        eta = 1.0
        le, dle = sq.L2(eta)
        lp_e, _ = sq.L1(eta)
        for ia, xi in enumerate(sq.GP3):
            lx, dlx = sq.L2(xi)
            lp_x, _ = sq.L1(xi)
            N, dxi, deta = [], [], []
            for bb in range(3):
                for aa in range(3):
                    N.append(lx[aa] * le[bb])
                    dxi.append(dlx[aa] * le[bb])
                    deta.append(lx[aa] * dle[bb])
            N = np.asarray(N); dxi = np.asarray(dxi); deta = np.asarray(deta)
            J2 = np.array([[dxi @ Xe[:, 0], deta @ Xe[:, 0]],
                           [dxi @ Xe[:, 1], deta @ Xe[:, 1]]])
            DG = np.column_stack([dxi, deta]) @ np.linalg.inv(J2)
            val = N @ ve
            der = np.einsum("ia,ic->ca", DG, ve)
            rr = val[0]
            Rq = N @ Xe[:, 0]
            F = np.array([[der[0, 0], 0.0, der[0, 1]],
                          [rr * der[1, 0], rr / Rq, rr * der[1, 1]],
                          [der[2, 0], 0.0, der[2, 1]]])
            J = np.linalg.det(F)
            Npv = np.array([lp_x[0] * lp_e[0], lp_x[1] * lp_e[0],
                            lp_x[1] * lp_e[1], lp_x[0] * lp_e[1]])
            pq = Npv @ pe
            P = piso(F, f, c, a, b) + pq * np.linalg.inv(F).T
            sig = (P @ F.T) / J * G0 * 1000.0
            dR = dxi @ Xe[:, 0]
            drdR = der[0, 0]
            aw = 2.0 * math.pi * rr * drdR * dR * sq.GW3[ia]
            out["r_cur"].append(rr); out["a_cur"].append(aw); out["szz_Pa"].append(sig[2, 2])
    return {k: np.asarray(v) for k, v in out.items()}
