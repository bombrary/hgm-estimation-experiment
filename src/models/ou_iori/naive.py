from tqdm import tqdm
from . import phi
from multiprocessing import Pool

def estimate(mu0, sig0, ys, model):
    mu = mu0
    sig = sig0

    mus = []
    sigs = []

    for y in tqdm(ys):
        phi0 = phi.phi0(y, mu, sig, model)
        phi1 = phi.phi1(y, mu, sig, model)
        phi2 = phi.phi2(y, mu, sig, model)

        mu = phi1 / phi0
        sig = phi2 / phi0 - mu**2

        mus.append(mu)
        sigs.append(sig)

    return mus, sigs
