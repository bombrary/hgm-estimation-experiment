import numpy as np
from scipy import stats, integrate
from . import Model
import multiprocessing
from numpy.typing import NDArray
from hgm_estimation.utils import derivative


DERIV_ORD = [ [1, 0, 1]
            , [0, 1, 1]
            , [0, 0, 2]
            , [1, 0, 0]
            , [0, 1, 0]
            , [0, 0, 1]
            , [0, 0, 0]
            ]


def p_gauss(xp: float, mu: float, lam: float) -> NDArray[np.float64]:
    return stats.norm.pdf(xp, loc=mu, scale=np.sqrt(1/lam))


def p_obs(y: float, x: float, *, model: Model) -> NDArray[np.float64]:
    return stats.norm.pdf(y, loc=(2*x)/(1+x**2), scale=np.sqrt(model.var_ob))


def p_st(x: float, xp: float, *, model: Model) -> NDArray[np.float64]:
    return stats.norm.pdf(x, loc=model.k*xp, scale=np.sqrt(model.var_st))


def p_mul(x: float, xp: float, y: float, mu: float, lam: float, *, model: Model) -> NDArray[np.float64]:
    return p_gauss(xp, mu, lam) * p_obs(y, x, model=model) * p_st(x, xp, model=model)


def phi0(y, mu, lam, *, model: Model) -> float:
    return integrate.dblquad(
        lambda x, xp, y, mu, lam: p_mul(x, xp, y, mu, lam, model=model),
        -np.inf, np.inf,
        lambda _: -np.inf, lambda _: np.inf, 
        args=(y, mu, lam)
    )[0]


def phi1(y, mu, lam, *, model: Model) -> float:
    return integrate.dblquad(
        lambda x, xp, y, mu, lam: x * p_mul(x, xp, y, mu, lam, model=model),
        -np.inf, np.inf,
        lambda _: -np.inf, lambda _: np.inf, 
        args=(y, mu, lam)
    )[0]


def phi2(y, mu, lam, *, model: Model) -> float:
    return integrate.dblquad(
        lambda x, xp, y, mu, lam: x*x * p_mul(x, xp, y, mu, lam, model=model),
        -np.inf, np.inf,
        lambda _: -np.inf, lambda _: np.inf, 
        args=(y, mu, lam)
    )[0]


def derivative_with_model(func, zs: list[float], ord: list[int], model: Model) -> float:
    def new_func(y, mu, lam):
        print(f"[DEBUG] func({y}, {mu}, {lam})")
        return func(y, mu, lam, model=model)

    return derivative(new_func, zs, ord)


def deriv_wrapper(t):
    return derivative_with_model(t[0], t[1], t[2], t[3])


def split_by_len(l, *len_list):
    res = []

    cur = 0
    for n in len_list:
        res.append(l[cur:cur+n])
        cur += n

    return res


def v_phis(y, mu, lam, *, model: Model):
    zs = [y, mu, lam]

    args0 = [(phi0, zs, order, model) for order in DERIV_ORD]
    args1 = [(phi1, zs, order, model) for order in DERIV_ORD]
    args2 = [(phi2, zs, order, model) for order in DERIV_ORD]
    args = [*args0, *args1, *args2]

    with multiprocessing.Pool(processes=10) as pool:
        r = pool.map(deriv_wrapper, args)

    return split_by_len(r, len(args0), len(args1), len(args2))


# model = Model(1/20, 1, 10)
# v_phis(0.01, 0.01, 1.0, model=model)

# Computation time: 2min (Arch Linux, Ryzen 7 5800X)
# [[2.7755575615628914e-05,
#   1.3877787807814457e-05,
#   0.0001942890293094024,
#   -5.9331914381566264e-05,
#   5.5963240985779095e-05,
#   -4.120781493810455e-05,
#   0.1222089207156917],
#  [-0.001928904085068961,
#   -0.0007298849025172416,
#   6.071532165918825e-05,
#   0.010759901400208757,
#   0.11641007604471909,
#   -2.6628005789924858e-05,
#   0.0012717074109904979],
#  [7.632783294297951e-05,
#   -4.163336342344337e-05,
#   0.22523649612082863,
#   -3.573889795216445e-05,
#   0.0024209494825511158,
#   -0.11216354085796798,
#   0.2272033734261355]]

# Warning come out
# 
# /home/bombrary/.pyenv/versions/3.10.5/lib/python3.10/site-packages/scipy/integrate/_quadpack_py.py:879: IntegrationWarning: The inte gral is probably divergent, or slowly convergent.
#   quad_r = quad(f, low, high, args=args, full_output=self.full_output,
