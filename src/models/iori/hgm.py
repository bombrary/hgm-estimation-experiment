import numpy as np
from tqdm import tqdm
from models.iori.iori_initial import v_phis_cache
from models.iori import pfaffian
from hgm_estimation import hgm


def create_z0(y1, mu1):
    y0 = 0.01 if y1 > 0 else -0.01
    mu0 = 0.01 if mu1 > 0 else -0.01
    return np.array([y0, mu0, 1.0])


def estimate(mu0, sig0, ys):
    # y = z0[0]
    mu = mu0
    sig = sig0

    mus = []
    sigs = []
    pbar = tqdm(ys)
    for y in pbar:

        z1 = np.array([y, mu, sig])
        z0 = create_z0(z1[0], z1[1])

        # tqdm.write(f'{z0} -> {z1}')

        v_phi00, v_phi10, v_phi20 = v_phis_cache(*z0)
        r0 = hgm.solve(z0, z1, v_phi00, lambda zs: pfaffian.phi0(zs))
        r1 = hgm.solve(z0, z1, v_phi10, lambda zs: pfaffian.phi1(zs))
        r2 = hgm.solve(z0, z1, v_phi20, lambda zs: pfaffian.phi2(zs))

        v_phi0 = r0.y[:, -1]
        v_phi1 = r1.y[:, -1]
        v_phi2 = r2.y[:, -1]

        # tqdm.write(f'v_phi0 = {list(v_phi0)}')
        # tqdm.write(f'v_phi1 = {list(v_phi1)}')
        # tqdm.write(f'v_phi2 = {list(v_phi2)}')

        mu = v_phi1[-1] / v_phi0[-1]
        sig = v_phi2[-1] / v_phi0[-1] - mu**2

        mus.append(mu)
        sigs.append(sig)

    return mus, sigs

