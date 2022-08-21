import numpy as np
from . import Model


def estimate_step(y, mu0, s0, model):
    k = model.k
    var_st = model.var_st
    l = model.l

    var_ob = model.var_ob
    s_pred = var_st + s0 * k**2
    mu_pred = k * mu0

    mu = (mu_pred*var_ob + s_pred*l*y) / (s_pred*l**2 + var_ob)
    s = (s_pred*l**2 + var_ob) / (s_pred*var_ob)

    return mu, s


def estimate(mu0, s0, ys, *, model: Model):

    mu = mu0
    s = s0

    mus = []
    ss = [] # NOTE: s0 = sig0^2
    for y in ys:
        mu, s = estimate_step(y, mu, s, model)

        mus.append(mu)
        ss.append(s)

    return np.array([mus, ss])
