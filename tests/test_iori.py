from hgm_estimation.utils import derivative1, derivative2
from hgm_estimation import hgm
from scipy import integrate, stats
import numpy as np
from multiprocessing import Pool
import pytest


def pfs_phi(c, i, zs):
    y, mu, sig = zs
    c.execute_string(f'subst(Pf{i}, y, {y}, mu, {mu}, sig, {sig});')
    return np.array(c.pop_cmo())

def p_obs(x, y):
    return stats.norm.pdf(y, loc=2*x/(1+x**2), scale=1)

def p_st(x, xp):
    return stats.norm.pdf(x, loc=4/5*xp, scale=1)

def p_mul(x, xp, y, mu, sig):
    a = p_st(x, xp)
    b = p_obs(x, y)
    c = stats.norm.pdf(xp, loc=mu, scale=np.sqrt(sig))
    return a*b*c


def dblquad_inf(fun, args):
    return integrate.dblquad(
            fun,
            -np.inf, np.inf,
            lambda _: -np.inf, lambda _: np.inf,
            args = args
           )[0]


def phi0(y, mu, sig):
    return dblquad_inf(p_mul, (y, mu, sig))


DERIV_ORD = [ [1, 0, 1]
            , [0, 1, 1]
            , [0, 0, 2]
            , [1, 0, 0]
            , [0, 1, 0]
            , [0, 0, 1]
            , [0, 0, 0]
            ]


def v_phi(fun, y, mu, sig, ord):
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
            return derivative1(lambda     sig: fun(y, mu, sig),        mu, 1)
        case [0, 0, 0]:
            return fun(y, mu, sig)
        case _:
            raise NotImplementedError()


def v_phis(y, mu, sig):
    args = [(phi0, y, mu, sig, ord) for ord in DERIV_ORD]

    with Pool(processes=12) as p:
        r = p.starmap(v_phi, args)

    return np.array(r, dtype=np.float64)


def test_phi0(client):
    z0 = np.array([0.01, 0.01, 1.0])
    # v_phi00 = v_phis(*z0)
    v_phi00 = np.array([-2.26946961e-04
                       ,-1.82199367e-04
                       ,3.10209292e-03
                       ,-8.15620917e-05
                       ,1.12648231e-03
                       ,-6.78056894e-03
                       ,2.93652370e-01])

    z1 = np.array([1.0, 2.0, 3.0])
    lhs = hgm.solve(z0, z1, v_phi00, lambda zs: pfs_phi(client, 0, zs)).y[:, -1]
    # rhs = v_phis(*z1)
    rhs = np.array([-0.005626
                   ,-0.00073575
                   ,0.00229936
                   ,-0.11128256
                   ,0.03087006
                   ,-0.01246692
                   ,0.32661882])

    assert pytest.approx(lhs, abs=0.01) == rhs
        

@pytest.mark.parametrize(('z0', 'z1'), [
    ([0.01, 0.02], [2, 1]),
    ([0.01, 0.02], [3, 1]),
    ([0.01, 0.02], [1, 2]),
])
def test_obs(client, z0, z1):
    client.load("tests/test_asir/iori-obs.rr");

    def fun(x, y):
        client.execute_string(f"matrix_matrix_to_list(subst(Pf, x, {x}, y, {y}));")
        return np.array(client.pop_cmo())

    obs0 = p_obs(*z0)
    obs1_actual = hgm.solve(np.array(z0), np.array(z1), np.array([obs0]), lambda zs: fun(*zs)).y[:, -1][0]
    obs1_desired = p_obs(*z1)

    assert pytest.approx(obs1_actual, abs=0.01) == obs1_desired


@pytest.mark.parametrize(('z0', 'z1'), [
    ([0.01, 0.02], [2, 1]),
    ([0.01, 0.02], [3, 1]),
    ([0.01, 0.02], [1, 2]),
])
def test_st(client, z0, z1):
    client.load("tests/test_asir/iori-st.rr");

    def fun(xx, x):
        client.execute_string(f"matrix_matrix_to_list(subst(Pf, xx, {xx}, x, {x}));")
        return np.array(client.pop_cmo())

    st0 = p_st(*z0)
    st1_actual = hgm.solve(np.array(z0), np.array(z1), np.array([st0]), lambda zs: fun(*zs)).y[:, -1][0]
    st1_desired = p_st(*z1)

    assert pytest.approx(st1_actual, abs=0.01) == st1_desired
