import numpy as np
from scipy.stats import norm
from scipy.integrate import dblquad
from scipy.misc import derivative
from multiprocessing import Pool

def S_sfkpp(xp):
    return 7/9*xp

def T_sfkpp(xp):
    return 1/8*xp - 1/9*xp**2

def p_ob(x, y):
    return float(norm.pdf(y, loc=x, scale=y))

def p_st(x, xp):
    loc = S_sfkpp(xp)
    scale = np.sqrt(T_sfkpp(xp)) # NOTE: scale is std, not variance
    return float(norm.pdf(x, loc=loc, scale=scale))

def p_postprev(xp, mu, sig):
    scale = np.sqrt(sig) # NOTE: Here sig is assumed variance, not std.
    return float(norm.pdf(xp, loc=mu, scale=scale))

def p_joint0(x, xp, y, mu, sig):
    return p_ob(x, y) * p_st(x, xp) * p_postprev(xp, mu, sig)

def p_joint1(x, xp, y, mu, sig):
    return x * p_ob(x, y) * p_st(x, xp) * p_postprev(xp, mu, sig)

def p_joint2(x, xp, y, mu, sig):
    return x * x * p_ob(x, y) * p_st(x, xp) * p_postprev(xp, mu, sig)

def dblquad01(fun, args):
    return dblquad(fun, 0, 1, lambda _: 0, lambda _: 1, args=args)[0]

def phi0(y, mu, sig):
    return dblquad01(p_joint0, (y, mu, sig))

def phi1(y, mu, sig):
    return dblquad01(p_joint1, (y, mu, sig))

def phi2(y, mu, sig):
    return dblquad01(p_joint2, (y, mu, sig))

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
    y0, mu0, sig0 = args
    # order = [y, mu, sig]
    match ns:
        case [0, 0, 0]:
            return func(y0, mu0, sig0)
        case [0, 0, n]:
            order = deriv_ord(n)
            return derivative(lambda sig: func(y0, mu0, sig), sig0, dx=dx, n=n, order=order)
        case [0, n, 0]:
            order = deriv_ord(n)
            return derivative(lambda mu: func(y0, mu, sig0), mu0, dx=dx, n=n, order=order)
        case [n, 0, 0]:
            order = deriv_ord(n)
            return derivative(lambda y: func(y, mu0, sig0), y0, dx=dx, n=n, order=order)
        case [1, 0, 1]:
            return derivative_11(lambda y, sig: func(y, mu0, sig), [y0, sig0], dx)
        case [0, 1, 1]:
            return derivative_11(lambda mu, sig: func(y0, mu, sig), [mu0, sig0], dx)
        case _:
            raise NotImplementedError()


# Standard monomials computed by Risa/Asir
# phi0: [dsig^3,dy^2,dsig*dy,dsig*dmu,dsig^2,dy,dmu,dsig,1]
# phi1: [dmu*dy,dsig*dy,dsig*dmu,dsig^2,dy,dmu,dsig,1]
# phi2: [dmu*dy,dsig*dy,dsig*dmu,dsig^2,dy,dmu,dsig,1]

def v_phi0(y, mu, sig):
    orders = [(0, 0, 3),
              (2, 0, 0),
              (1, 0, 1),
              (0, 1, 1),
              (0, 0, 2),
              (1, 0, 0),
              (0, 1, 0),
              (0, 0, 1),
              (0, 0, 0)]

    args = [(phi0, (y, mu, sig), order) for order in orders]

    with Pool(processes=12) as p:
        r = p.starmap(my_derivative, args)

    return np.array(r, dtype=np.float64)
