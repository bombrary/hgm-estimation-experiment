import numpy as np
from tqdm import tqdm
from hgm_estimation import hgm

def estimate(mu0, sig0, ys, *,
             fun_z0_vphi0,
             fun_z0_vphi1,
             fun_z0_vphi2,
             pfs_phi0,
             pfs_phi1,
             pfs_phi2,
             event_phi0 = None,
             event_phi1 = None,
             event_phi2 = None,
             log=False,
             disable_tqdm=False):
    mu = mu0
    sig = sig0

    mus = []
    sigs = []

    for y in tqdm(ys, disable=disable_tqdm):
        z1 = np.array([y, mu, sig])

        z00, v_phi00 = fun_z0_vphi0(z1)
        z10, v_phi10 = fun_z0_vphi1(z1)
        z20, v_phi20 = fun_z0_vphi2(z1)

        z00 = np.asarray(z00)
        z10 = np.asarray(z10)
        z20 = np.asarray(z20)

        if event_phi0 is not None:
            event_phi0.c = lambda t: (1-t)*z00 + t*z1

        if event_phi1 is not None:
            event_phi1.c = lambda t: (1-t)*z10 + t*z1

        if event_phi2 is not None:
            event_phi2.c = lambda t: (1-t)*z20 + t*z1

        if log:
            tqdm.write(f'{list(z00)}, {list(z10)}, {list(z20)} -> {list(z1)}')
        r0 = hgm.solve(z00, z1, v_phi00, lambda zs: pfs_phi0(zs), events=event_phi0)
        r1 = hgm.solve(z10, z1, v_phi10, lambda zs: pfs_phi1(zs), events=event_phi1)
        r2 = hgm.solve(z20, z1, v_phi20, lambda zs: pfs_phi2(zs), events=event_phi2)

        v_phi0 = r0.y[:, -1]
        v_phi1 = r1.y[:, -1]
        v_phi2 = r2.y[:, -1]

        mu = v_phi1[-1] / v_phi0[-1]
        sig = v_phi2[-1] / v_phi0[-1] - mu**2

        mus.append(mu)
        sigs.append(sig)

    return mus, sigs

