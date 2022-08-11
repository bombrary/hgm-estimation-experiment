from tqdm import tqdm
from .phi import v_phis_analytic


def run_naive(y0, mu0, lam0, ys, *, model):

    y, mu, lam = y0, mu0, lam0

    mus = []
    lams = []

    pbar = tqdm(ys)
    for y in pbar:
        v_phi0, v_phi1, v_phi2 =  v_phis_analytic(y, mu, lam, model=model)


        # NOTE: The **LAST** index of standard monomial is 1.
        # So we have to take the last columns of v_phi*.
        mu = v_phi1[-1] / v_phi0[-1]

        sig_sq = v_phi2[-1] / v_phi0[-1] - mu**2
        lam = 1 / sig_sq  # note: lam = 1 / sigma

        mus.append(mu)
        lams.append(lam)

    return mus, lams
