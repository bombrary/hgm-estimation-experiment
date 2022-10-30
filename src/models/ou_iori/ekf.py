import numpy as np
from . import Model


def estimate_step(y, mu0, s0, model: Model):
    k = model.k
    var_st = model.var_st
    var_ob = model.var_ob

    # NOTE:
    #   state: x = k*xp + v,    v ~ N(0, var_st)
    # observe: y = l*y + u + w, w ~ N(0, var_ob)
    # l and u is derived from derivation of 2*x/(1 + x**2).

    s_pred = var_st + s0 * k**2
    mu_pred = k * mu0

    l = -2*(mu_pred - 1) / (1 + mu_pred**2)**2
    u = 2*mu_pred**2/(1 + mu_pred**2)

    mu = (mu_pred*var_ob + s_pred*l*(y-u)) / (s_pred*l**2 + var_ob)
    s = (s_pred*l**2 + var_ob) / (s_pred*var_ob)

    return mu, s


def estimate(mu0, s0, ys, model: Model):
    mu = mu0
    s = s0

    mus = []
    ss = [] # NOTE: s0 = sig0^2
    for y in ys:
        mu, s = estimate_step(y, mu, s, model)

        mus.append(mu)
        ss.append(s)

    return np.array([mus, ss])
