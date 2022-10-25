import numpy as np
from hgm_estimation import hgm
from ..linear.phi import v_phis_analytic
from tqdm import tqdm

def estimate(y0, mu0, sig0, ys, *,
        v_phi00, v_phi10, v_phi20,
        pfs_phi0, pfs_phi1, pfs_phi2,
        model):
    mu = mu0
    sig = sig0
    z0 = np.array([y0, mu0, sig0])

    mus = []
    sigs = []

    for y in tqdm(ys):
        z1 = np.array([y, mu, sig])

        r0 = hgm.solve(z0, z1, v_phi00, lambda zs: pfs_phi0(zs))
        r1 = hgm.solve(z0, z1, v_phi10, lambda zs: pfs_phi1(zs))
        r2 = hgm.solve(z0, z1, v_phi20, lambda zs: pfs_phi2(zs))

        v_phi0 = r0.y[:, -1]
        v_phi1 = r1.y[:, -1]
        v_phi2 = r2.y[:, -1]

        mu = v_phi1[-1] / v_phi0[-1]
        sig = v_phi2[-1] / v_phi0[-1] - mu**2

        mus.append(mu)
        sigs.append(sig)

    return mus, sigs

