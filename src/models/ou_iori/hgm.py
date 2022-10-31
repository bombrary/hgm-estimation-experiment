import numpy as np
from tqdm import tqdm
from hgm_estimation import hgm

def estimate(mu0, sig0, ys, *,
             fun_z0_vphi0,
             fun_z0_vphi1,
             fun_z0_vphi2,
             pfs_phi0, pfs_phi1, pfs_phi2, log=False, disable_tqdm=False):
    mu = mu0
    sig = sig0

    mus = []
    sigs = []

    for y in tqdm(ys, disable=disable_tqdm):
        z1 = np.array([y, mu, sig])

        z00, v_phi00 = fun_z0_vphi0(z1)
        z10, v_phi10 = fun_z0_vphi1(z1)
        z20, v_phi20 = fun_z0_vphi2(z1)

        if log:
            tqdm.write(f'{z00}, {z10}, {z20} -> {z1}')
        r0 = hgm.solve(z00, z1, v_phi00, lambda zs: pfs_phi0(zs))
        r1 = hgm.solve(z10, z1, v_phi10, lambda zs: pfs_phi1(zs))
        r2 = hgm.solve(z20, z1, v_phi20, lambda zs: pfs_phi2(zs))

        v_phi0 = r0.y[:, -1]
        v_phi1 = r1.y[:, -1]
        v_phi2 = r2.y[:, -1]

        mu = v_phi1[-1] / v_phi0[-1]
        sig = v_phi2[-1] / v_phi0[-1] - mu**2

        mus.append(mu)
        sigs.append(sig)

    return mus, sigs

