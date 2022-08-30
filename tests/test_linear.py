import pytest
from hgm_estimation import hgm
from src.models.linear.phi import v_phis_analytic
from src.models.linear import Model
from src.utils import pfs_phi
from multiprocessing import Pool
from scipy import stats, integrate
import numpy as np

def p_mul(x, xp, y, mu, lam, model: Model):
    std_ob = np.sqrt(model.var_ob)
    std_st = np.sqrt(model.var_st)
    sig = np.sqrt(1/lam)
    k = model.k
    l = model.l

    p_obs = lambda x, y: stats.norm.pdf(y, loc=l*x, scale=std_ob)
    p_st = lambda x, xp: stats.norm.pdf(x, loc=k*xp, scale=std_st)
    p_gauss = lambda xp: stats.norm.pdf(xp, loc=mu, scale=sig)

    return p_obs(x, y) * p_st(x, xp) * p_gauss(xp)


def dblquad_inf(fun):
    return integrate.dblquad(
            fun,
            -np.inf, np.inf,
            lambda _: -np.inf, lambda _: np.inf)[0]



def phi0(y, mu, lam, model: Model):
    fun = lambda x, xp: p_mul(x, xp, y, mu, lam, model)
    return dblquad_inf(fun)


def phi1(y, mu, lam, model: Model):
    fun = lambda x, xp: x * p_mul(x, xp, y, mu, lam, model)
    return dblquad_inf(fun)


def phi2(y, mu, lam, model: Model):
    fun = lambda x, xp: x * x * p_mul(x, xp, y, mu, lam, model)
    return dblquad_inf(fun)


def phi_numeric(fun, y, mu, lam, model: Model):
    return fun(y, mu, lam, model)


def gen_pfaffian(client, model: Model):
    k, var_st, l, var_ob = model.to_frac()
    client.load("asir-src/phi-linear.rr")
    client.execute_string(f'Pfs = gen_pfaffian({k}, {var_st}, {l}, {var_ob});')


@pytest.mark.parametrize(('model', 'z'), [
    (Model(1.0, 1.0, 1.0, 1.0), [0.01, 0.01, 1.0]),
    (Model(1.0, 1/10, 1.0, 1.0), [0.01, 0.01, 1.0]),
    (Model(0.1, 1.0, 1.0, 1.0), [0.01, 0.01, 1.0]),
])
def test_v_phis_analytic(model, z):
    phis_desired = [e[0] for e in v_phis_analytic(*z, model=model)]

    with Pool(processes = 3) as p:
        phis_actual = p.starmap(phi_numeric,
                     [(phi0, *z, model)
                     ,(phi1, *z, model)
                     ,(phi2, *z, model)])

    assert pytest.approx(phis_desired, abs=0.01) == phis_actual


@pytest.mark.parametrize(('model', 'z0', 'z1'), [
    (Model(1.0, 1.0, 1.0, 1.0), [0.01, 0.01, 1.0], [1, 2, 3]),
    (Model(1.0, 1/10, 1.0, 1.0), [0.01, 0.01, 1.0], [1, 2, 3]),
])
def test_hgm(client, model, z0, z1):
    gen_pfaffian(client, model)

    y0, y1, y2 = v_phis_analytic(*z0, model=model)

    phi0 = hgm.solve(np.array(z0), np.array(z1), y0, lambda zs: pfs_phi(client, 0, zs)).y[:, -1]
    phi1 = hgm.solve(np.array(z0), np.array(z1), y1, lambda zs: pfs_phi(client, 1, zs)).y[:, -1]
    phi2 = hgm.solve(np.array(z0), np.array(z1), y2, lambda zs: pfs_phi(client, 2, zs)).y[:, -1]

    phi0_desired, phi1_desired, phi2_desired = list(v_phis_analytic(*z1, model=model))

    assert phi0[0] == pytest.approx(phi0_desired[0], abs=1e-2)
    assert phi1[0] == pytest.approx(phi1_desired[0], abs=1e-2)
    assert phi2[0] == pytest.approx(phi2_desired[0], abs=1e-2)
