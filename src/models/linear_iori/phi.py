from scipy import integrate, stats
import numpy as np
from hgm_estimation.utils import derivative1, derivative2
from multiprocessing import Pool
from . import Model


def p_obs(x, y, model: Model):
    var = model.var_ob
    return float(stats.norm.pdf(y, loc=2*x/(1+x**2), scale=np.sqrt(var)))


def p_st(x, xp, model: Model):
    var = model.var_st
    k = model.k
    return float(stats.norm.pdf(x, loc=k*xp, scale=np.sqrt(var)))


def p_gauss(xp, mu, sig):
    return float(stats.norm.pdf(xp, loc=mu, scale=np.sqrt(sig)))


def p_pred(x, mu, sig, model):
    return float(stats.norm.pdf(x, loc=model.k*mu, scale=np.sqrt(model.k**2*sig + model.var_st)))


def p_mul(x, xp, y, mu, sig, model):
    return p_st(x, xp, model) * p_obs(x, y, model) * p_gauss(xp, mu, sig)


def dblquad_inf(fun, args):
    return integrate.dblquad(
            fun,
            -np.inf, np.inf,
            lambda _: -np.inf, lambda _: np.inf,
            args = args
           )[0]


def quad_inf(fun):
    return integrate.quad(fun, -np.inf, np.inf)[0]


def phi0(y, mu, sig, model: Model):
    fun = lambda x: p_obs(x, y, model) * p_pred(x, mu, sig, model)
    return quad_inf(fun)


def phi1(y, mu, sig, model: Model):
    fun = lambda x: x * p_obs(x, y, model) * p_pred(x, mu, sig, model)
    return quad_inf(fun)


def phi2(y, mu, sig, model: Model):
    fun = lambda x: x * x * p_obs(x, y, model) * p_pred(x, mu, sig, model)
    return quad_inf(fun)


DERIV_ORD = [ [1, 0, 1]
            , [0, 1, 1]
            , [0, 0, 2]
            , [1, 0, 0]
            , [0, 1, 0]
            , [0, 0, 1]
            , [0, 0, 0]
            ]


# [dsig*dy,dsig*dmu,dsig^2,dy,dmu,dsig,1]
def phi_deriv(fun, y, mu, sig, ord, model: Model, debug=True):
    if debug:
        print(f'[DEBUG] {fun}({y}, {mu}, {sig}, {ord})')
    match ord:
        case [1, 0, 1]:
            return derivative2(lambda  y, sig: fun(y, mu, sig, model),  [y, sig], [1, 1])
        case [0, 1, 1]:
            return derivative2(lambda mu, sig: fun(y, mu, sig, model), [mu, sig], [1, 1])
        case [0, 0, 2]:
            return derivative1(lambda     sig: fun(y, mu, sig, model),       sig, 2)
        case [1, 0, 0]:
            return derivative1(lambda       y: fun(y, mu, sig, model),         y, 1)
        case [0, 1, 0]:
            return derivative1(lambda      mu: fun(y, mu, sig, model),        mu, 1)
        case [0, 0, 1]:
            return derivative1(lambda     sig: fun(y, mu, sig, model),        sig, 1)
        case [0, 0, 0]:
            return fun(y, mu, sig, model)
        case _:
            raise NotImplementedError()


def v_phi(phi, y, mu, sig, model: Model):
    args = [(phi, y, mu, sig, ord, model) for ord in DERIV_ORD]

    with Pool(processes=7) as p:
        r = p.starmap(phi_deriv, args)

    return r


def v_phi2(phi, z0, z1, model: Model):
    args0 = [(phi, *z0, ord, model) for ord in DERIV_ORD]
    args1 = [(phi, *z1, ord, model) for ord in DERIV_ORD]
    args = [*args0, *args1]

    with Pool(processes=12) as p:
        r = p.starmap(phi_deriv, args)
        r1 = r[:7]
        r2 = r[7:14]

    return r1, r2


def v_phis(y, mu, sig, model: Model, debug=True, processes=12):
    args0 = [(phi0, y, mu, sig, ord, model, debug) for ord in DERIV_ORD]
    args1 = [(phi1, y, mu, sig, ord, model, debug) for ord in DERIV_ORD]
    args2 = [(phi2, y, mu, sig, ord, model, debug) for ord in DERIV_ORD]
    args = [*args0, *args1, *args2]

    with Pool(processes=processes) as p:
        r = p.starmap(phi_deriv, args)
        r1 = r[:7]
        r2 = r[7:14]
        r3 = r[14:]

    return r1, r2, r3
