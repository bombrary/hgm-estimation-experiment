from tqdm import tqdm
from multiprocessing import Pool
from models.iori.phi import phi0, phi1, phi2


def phi_wrapper(fun, y, mu, sig):
    return fun(y, mu, sig)


def estimate(mu0, sig0, ys):

    mu, sig = mu0, sig0

    mus = []
    sigs = []

    pbar = tqdm(ys)
    for y in pbar:
        # tqdm.write(f'mu,sig = {mu,sig}')
        with Pool(processes=3) as p:
            args = [(phi0, y, mu, sig)
                   ,(phi1, y, mu, sig)
                   ,(phi2, y, mu, sig)
                   ]
            p0, p1, p2 = p.starmap(phi_wrapper, args)


        mu = p1 / p0
        sig = p2 / p0 - mu**2

        mus.append(mu)
        sigs.append(sig)

    return mus, sigs
