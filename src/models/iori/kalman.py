import numpy as np


def estimate_step(y, mu0, s0, k, var_st, var_ob):
    s_pred = var_st + s0 * k**2
    mu_pred = k * mu0

    l = 2*(mu_pred - 1) / (1 + mu_pred**2)**2
    u = 2*mu_pred**2/(1 + mu_pred**2)

    mu = (mu_pred*var_ob + s_pred*l*(y-u)) / (s_pred*l**2 + var_ob)
    s = (s_pred*l**2 + var_ob) / (s_pred*var_ob)

    return mu, s


def estimate(mu0, s0, ys, *, k, var_st, var_ob):

    mu = mu0
    s = s0

    mus = []
    ss = [] # NOTE: s0 = sig0^2
    for y in ys:
        mu, s = estimate_step(y, mu, s, k, var_st, var_ob)

        mus.append(mu)
        ss.append(s)

    return np.array([mus, ss])
