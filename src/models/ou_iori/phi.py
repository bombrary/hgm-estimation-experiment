import numpy as np
from scipy import stats, integrate
from scipy.misc import derivative
from . import Model
from multiprocessing import Pool
from numpy.typing import NDArray

def p_st(x, xp, model: Model):
    return float(stats.norm.pdf(x, loc=model.k*xp, scale=np.sqrt(model.var_st)))


def p_ob(x, y, model: Model):
    return float(stats.norm.pdf(y, loc=2*x/(1 + x**2), scale=np.sqrt(model.var_ob)))


def p_postprev(xp, mu, sig):
    # NOTE: Here sig is a variance, not a std.
    return float(stats.norm.pdf(xp, loc=mu, scale=sig))


def p_pred(x, mu, sig, model):
    # Compute "∫ p_st * p_posrprev dx"
    # In this model, it can be computed analytically
    loc = model.k * mu
    scale = np.sqrt(sig*model.k**2 + 1**2) # NOTE: scale is std, not variance.
    return float(stats.norm.pdf(x, loc=loc, scale=scale))


def p_join0(x, y, mu, sig, model):
    return p_pred(x, mu, sig, model) * p_ob(x, y, model)


def p_join1(x, y, mu, sig, model):
    return x * p_join0(x, y, mu, sig, model)


def p_join2(x, y, mu, sig, model):
    return x * x * p_join0(x, y, mu, sig, model)


def quad_inf(func, args):
    return integrate.quad(func,
                          -np.inf, np.inf,
                          args)[0]

def phi0(y, mu, sig, model):
    return quad_inf(p_join0, (y, mu, sig, model))


def phi1(y, mu, sig, model):
    return quad_inf(p_join1, (y, mu, sig, model))


def phi2(y, mu, sig, model):
    return quad_inf(p_join2, (y, mu, sig, model))


# Standard Monomials
# phi0: [dsig*dy,dsig*dmu,dsig^2,dy,dmu,dsig,1]
# phi1: [dsig*dy,dsig*dmu,dsig^2,dy,dmu,dsig,1]
# phi2: [dsig*dy,dsig*dmu,dsig^2,dy,dmu,dsig,1]

DERIV_ORD = [ [1, 0, 1]
            , [0, 1, 1]
            , [0, 0, 2]
            , [1, 0, 0]
            , [0, 1, 0]
            , [0, 0, 1]
            , [0, 0, 0]
            ]


# Compute d^2/(dx*dy) fun(x,y)
def derivative_11(fun, zs, dz=1e-3):
    a = fun(zs[0] + dz, zs[1] + dz)
    b = fun(zs[0] - dz, zs[1] + dz)
    c = fun(zs[0] + dz, zs[1] - dz)
    d = fun(zs[0] - dz, zs[1] - dz)
    return (a - b - c + d) / (4*dz**2)


def deriv_ord(n):
    if n % 2 == 0:
        return n + 1
    else:
        return n + 2


def my_derivative(func, args, ns, *, dx=1e-3):
    y0, mu0, sig0, model = args
    # order = [y, mu, sig]
    match ns:
        case [0, 0, 0]:
            return func(*args)
        case [0, 0, n]:
            order = deriv_ord(n)
            return derivative(lambda sig: func(y0, mu0, sig, model), sig0, dx=dx, n=n, order=order)
        case [0, n, 0]:
            order = deriv_ord(n)
            return derivative(lambda mu: func(y0, mu, sig0, model), mu0, dx=dx, n=n, order=order)
        case [n, 0, 0]:
            order = deriv_ord(n)
            return derivative(lambda y: func(y, mu0, sig0, model), y0, dx=dx, n=n, order=order)
        case [1, 0, 1]:
            return derivative_11(lambda y, sig: func(y, mu0, sig, model), [y0, sig0], dx)
        case [0, 1, 1]:
            return derivative_11(lambda mu, sig: func(y0, mu, sig, model), [mu0, sig0], dx)
        case _:
            raise NotImplementedError()


def v_phi(phi, y, mu, sig, model):
    args = [(phi, (y, mu, sig, model), ns) for ns in DERIV_ORD]

    with Pool(processes=12) as p:
        r = p.starmap(my_derivative, args)

    return np.array(r, dtype=np.float64)


def v_phis(y, mu, sig, model):
    args0 = [(phi0, (y, mu, sig, model), ns) for ns in DERIV_ORD]
    args1 = [(phi1, (y, mu, sig, model), ns) for ns in DERIV_ORD]
    args2 = [(phi2, (y, mu, sig, model), ns) for ns in DERIV_ORD]
    args = [*args0, *args1, *args2]

    N = len(DERIV_ORD)

    with Pool(processes=12) as p:
        r = p.starmap(my_derivative, args)
        v_phi0 = r[:N]
        v_phi1 = r[N:2*N]
        v_phi2 = r[2*N:]
        

    return np.array([v_phi0, v_phi1, v_phi2], dtype=np.float64)
