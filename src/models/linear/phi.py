import numpy as np
from scipy import stats
from . import Model


def phi_component(y, mu, lam, *, model: Model):
    k = model.k
    var_st = model.var_st
    l = model.l
    var_ob = model.var_ob

    s = 1/lam # s = sigma^2
    s_pred = var_st + s * k**2
    mu_pred = k * mu

    mu = (mu_pred * var_ob + s_pred * l * y) / (s_pred * l**2 + var_ob)
    s = (s_pred * var_ob) / (s_pred * l**2 + var_ob)
    
    coeff = stats.norm.pdf(y, loc=l*mu_pred, scale=np.sqrt(s_pred * l**2 + var_ob))

    return coeff, mu, s


def phi0_analytic(y, mu, lam, *, model):
    return phi_component(y, mu, lam, model=model)[0]


def phi1_analytic(y, mu, lam, *, model):
    coeff, mu_new, _ = phi_component(y, mu, lam, model=model)
    return coeff * mu_new


def phi2_analytic(y, mu, lam, *, model):
    coeff, mu_new, s_new = phi_component(y, mu, lam, model=model)
    return coeff * (mu_new**2 + s_new)


def v_phis_analytic(y, mu, lam, *, model):
    return [ np.array([phi0_analytic(y, mu, lam, model=model)])
           , np.array([phi1_analytic(y, mu, lam, model=model)])
           , np.array([phi2_analytic(y, mu, lam, model=model)])
           ]
