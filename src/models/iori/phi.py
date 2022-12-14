from scipy import integrate, stats
import numpy as np
from hgm_estimation.utils import derivative1, derivative2
from multiprocessing import Pool

def p_obs(x, y):
    return float(stats.norm.pdf(y, loc=2*x/(1+x**2), scale=1))


def p_st(x, xp):
    return float(stats.norm.pdf(x, loc=4/5*xp, scale=1))


def p_gauss(xp, mu, sig):
    return float(stats.norm.pdf(xp, loc=mu, scale=np.sqrt(sig)))


def p_mul(x, xp, y, mu, sig):
    return p_st(x, xp) * p_obs(x, y) * p_gauss(xp, mu, sig)


def p_pred(x, mu, sig):
    return float(stats.norm.pdf(x, loc=4/5*mu, scale=np.sqrt((4/5)**2*sig + 1)))


def dblquad_inf(fun):
    return integrate.dblquad(
            fun,
            -np.inf, np.inf,
            lambda _: -np.inf, lambda _: np.inf)[0]


def quad_inf(fun):
    return integrate.quad(fun, -np.inf, np.inf)[0]


def phi0(y, mu, sig):
    fun = lambda x: p_obs(x, y) * p_pred(x, mu, sig)
    return quad_inf(fun)


def phi1(y, mu, sig):
    fun = lambda x: x * p_obs(x, y) * p_pred(x, mu, sig)
    return quad_inf(fun)


def phi2(y, mu, sig):
    fun = lambda x: x * x * p_obs(x, y) * p_pred(x, mu, sig)
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
def phi_deriv(fun, y, mu, sig, ord):
    print(f'[DEBUG] {fun}({y}, {mu}, {sig}, {ord})')
    match ord:
        case [1, 0, 1]:
            return derivative2(lambda  y, sig: fun(y, mu, sig),  [y, sig], [1, 1])
        case [0, 1, 1]:
            return derivative2(lambda mu, sig: fun(y, mu, sig), [mu, sig], [1, 1])
        case [0, 0, 2]:
            return derivative1(lambda     sig: fun(y, mu, sig),       sig, 2)
        case [1, 0, 0]:
            return derivative1(lambda       y: fun(y, mu, sig),         y, 1)
        case [0, 1, 0]:
            return derivative1(lambda      mu: fun(y, mu, sig),        mu, 1)
        case [0, 0, 1]:
            return derivative1(lambda     sig: fun(y, mu, sig),        sig, 1)
        case [0, 0, 0]:
            return fun(y, mu, sig)
        case _:
            raise NotImplementedError()


def v_phi(phi, y, mu, sig):
    args = [(phi, y, mu, sig, ord) for ord in DERIV_ORD]

    with Pool(processes=12) as p:
        r = p.starmap(phi_deriv, args)

    return np.array(r, dtype=np.float64)


def v_phis(y, mu, sig):
    args0 = [(phi0, y, mu, sig, ord) for ord in DERIV_ORD]
    args1 = [(phi1, y, mu, sig, ord) for ord in DERIV_ORD]
    args2 = [(phi2, y, mu, sig, ord) for ord in DERIV_ORD]
    args = [*args0, *args1, *args2]

    with Pool(processes=12) as p:
        r = p.starmap(phi_deriv, args)
        r1 = r[:7]
        r2 = r[7:14]
        r3 = r[14:]

    return r1, r2, r3
