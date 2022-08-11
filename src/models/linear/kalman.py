import numpy as np
from . import Model


def estimate(mu0, s0, ys, *, model: Model):
    k = model.k
    var_st = model.var_st
    l = model.l
    var_ob = model.var_ob

    mus = [mu0]
    ss = [s0] # NOTE: s0 = sig0^2
    for y in ys:
        s_pred = var_st + ss[-1] * k**2
        mu_pred = k * mus[-1]

        mu = (mu_pred*var_ob + s_pred*l*y) / (s_pred*l**2 + var_ob)
        s = (s_pred*l**2 + var_ob) / (s_pred*var_ob)

        mus.append(mu)
        ss.append(s)

    return np.array([mus, ss])
