from hgm_estimation.utils import derivative1, derivative2
from hgm_estimation import hgm
from scipy import integrate, stats
import numpy as np
from multiprocessing import Pool
import pytest


def p_obs(x, y):
    return float(stats.norm.pdf(y, loc=2*x/(1+x**2), scale=1))


def p_st(x, xp):
    return float(stats.norm.pdf(x, loc=4/5*xp, scale=1))


def p_gauss(xp, mu, sig):
    return float(stats.norm.pdf(xp, loc=mu, scale=np.sqrt(sig)))


def p_mul(x, xp, y, mu, sig):
    return p_st(x, xp) * p_obs(x, y) * p_gauss(xp, mu, sig)


def dblquad_inf(fun, args):
    return integrate.dblquad(
            fun,
            -np.inf, np.inf,
            lambda _: -np.inf, lambda _: np.inf,
            args = args
           )[0]


def phi0(y, mu, sig):
    return dblquad_inf(p_mul, (y, mu, sig))


def phi1(y, mu, sig):
    fun = lambda x, xp, y, mu, sig: x * p_mul(x, xp, y, mu, sig)
    return dblquad_inf(fun, (y, mu, sig))


def phi2(y, mu, sig):
    fun = lambda x, xp, y, mu, sig: x * x * p_mul(x, xp, y, mu, sig)
    return dblquad_inf(fun, (y, mu, sig))


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
            return derivative1(lambda     sig: fun(y, mu, sig),        sig, 1)
        case [0, 0, 0]:
            return fun(y, mu, sig)
        case _:
            raise NotImplementedError()


def v_phis(phi, y, mu, sig):
    args = [(phi, y, mu, sig, ord) for ord in DERIV_ORD]

    with Pool(processes=12) as p:
        r = p.starmap(v_phi, args)

    return np.array(r, dtype=np.float64)


@pytest.mark.parametrize(('z0', 'z1'), [
    ([0.01, 0.02], [2, 1]),
    ([0.01, 0.02], [1, 2]),
])
def test_obs(client, z0, z1):
    client.load("tests/test_asir/iori-obs.rr");

    def fun(x, y):
        client.execute_string(f"matrix_matrix_to_list(subst(Pf, x, {x}, y, {y}));")
        return np.array(client.pop_cmo())

    val0 = p_obs(*z0)
    val1_actual = hgm.solve(z0, z1, [val0], lambda zs: fun(*zs)).y[:, -1][0]
    val1_desired = p_obs(*z1)

    assert pytest.approx(val1_actual, abs=0.01) == val1_desired


@pytest.mark.parametrize(('z0', 'z1'), [
    ([0.01, 0.02], [2, 1]),
    ([0.01, 0.02], [1, 2]),
])
def test_st(client, z0, z1):
    client.load("tests/test_asir/iori-st.rr");

    def fun(xx, x):
        client.execute_string(f"matrix_matrix_to_list(subst(Pf, xx, {xx}, x, {x}));")
        return np.array(client.pop_cmo())

    val0 = p_st(*z0)
    val1_actual = hgm.solve(z0, z1, [val0], lambda zs: fun(*zs)).y[:, -1][0]
    val1_desired = p_st(*z1)

    assert pytest.approx(val1_actual, abs=0.01) == val1_desired


@pytest.mark.parametrize(('z0', 'z1'), [
    ([0.01, 0.02, 1.0], [1, 2, 3]),
    ([0.01, 0.02, 1.0], [3, 2, 1]),
])
def test_gauss(client, z0, z1):
    client.load("tests/test_asir/iori-gauss.rr");

    def fun(x, mu, sig):
        client.execute_string(f"matrix_matrix_to_list(subst(Pf, x, {x}, mu, {mu}, sig, {sig}));")
        return np.array(client.pop_cmo())

    val0 = p_gauss(*z0)
    val1_actual = hgm.solve(z0, z1, [val0], lambda zs: fun(*zs)).y[:, -1][0]
    val1_desired = p_gauss(*z1)

    assert pytest.approx(val1_actual, abs=0.01) == val1_desired


@pytest.mark.parametrize(('z0', 'z1'), [
    ([0.01, 0.01, 0.02, 1.0], [1, 2, 1, 4]),
    ([0.01, 0.01, 0.02, 1.0], [1, 3, 1, 4]),
    ([0.01, 0.01, 0.02, 1.0], [1, 1, 2, 4]),
])
def test_mul_gauss_st(client, z0, z1):
    client.load("tests/test_asir/iori-mul.rr");

    def fun(xx, x, mu, sig):
        client.execute_string(f"subst(Pf1, xx, {xx}, x, {x}, mu, {mu}, sig, {sig});")
        return np.array(client.pop_cmo())

    def p(xx, x, mu, sig):
        return p_gauss(x, mu, sig) * p_st(xx, x)


    val0 = p(*z0)
    val1_actual = hgm.solve(z0, z1, [val0], lambda zs: fun(*zs)).y[:, -1][0]
    val1_desired = p(*z1)

    assert pytest.approx(val1_actual, abs=0.01) == val1_desired


@pytest.mark.parametrize(('z0', 'z1'), [
    ([0.02, 0.01, 0.01, 0.02, 1.0], [1, 2, 3, 4, 5]),
    ([0.02, 0.01, 0.01, 0.02, 1.0], [5, 4, 3, 2, 1]),
])
def test_mul(client, z0, z1):
    client.load("tests/test_asir/iori-mul.rr");

    def fun(xx, x, y, mu, sig):
        client.execute_string(f"subst(Pf2, xx, {xx}, x, {x}, y, {y}, mu, {mu}, sig, {sig});")
        return np.array(client.pop_cmo())

    p = p_mul

    val0 = p(*z0)
    val1_actual = hgm.solve(z0, z1, [val0], lambda zs: fun(*zs)).y[:, -1][0]
    val1_desired = p(*z1)

    assert pytest.approx(val1_actual, abs=0.01) == val1_desired


@pytest.mark.parametrize(('z0', 'z1'), [
    ([0.01, 0.01, 0.02, 1.0], [1, 2, 3, 4]),
    ([0.01, 0.01, 0.02, 1.0], [4, 3, 2, 1]),
])
def test_int1(client, z0, z1):
    client.load("tests/test_asir/iori-int1.rr");

    def fun(xx, y, mu, sig):
        client.execute_string(f"subst(Pf, xx, {xx}, y, {y}, mu, {mu}, sig, {sig});")
        return np.array(client.pop_cmo())

    def p(xx, y, mu, sig):
        return integrate.quad(lambda x: p_mul(xx, x, y, mu, sig)
                             ,-np.inf
                             ,np.inf
                             )[0]

    val0 = p(*z0)
    val1_actual = hgm.solve(z0, z1, [val0], lambda zs: fun(*zs)).y[:, -1][0]
    val1_desired = p(*z1)

    assert pytest.approx(val1_actual, abs=0.01) == val1_desired


@pytest.mark.parametrize(('z0', 'z1'), [
    ([0.01, 0.01, 0.02, 1.0], [1, 2, 3, 4]),
    ([0.01, 0.01, 0.02, 1.0], [4, 3, 2, 1]),
])
def test_int2(client, z0, z1):
    client.load("tests/test_asir/iori-int2.rr");

    def fun(x, y, mu, sig):
        client.execute_string(f"subst(Pf, x, {x}, y, {y}, mu, {mu}, sig, {sig});")
        return np.array(client.pop_cmo())

    def int_pmul(x, y, mu, sig):
        return integrate.quad(lambda xx: p_mul(xx, x, y, mu, sig)
                             ,-np.inf
                             ,np.inf
                             )[0]
    
    def p(x, y, mu, sig):
        return [ derivative1(lambda y: int_pmul(x, y, mu, sig), y, 3)
               , derivative1(lambda x: int_pmul(x, y, mu, sig), x, 2)
               , derivative2(lambda x, y: int_pmul(x, y, mu, sig), [x,y], [1,1])
               , derivative1(lambda y: int_pmul(x, y, mu, sig), y, 2)
               , derivative1(lambda x: int_pmul(x, y, mu, sig), x, 1)
               , derivative1(lambda y: int_pmul(x, y, mu, sig), y, 1)
               , int_pmul(x, y, mu, sig)
               ]


    val0 = p(*z0)
    val1_actual = hgm.solve(z0, z1, val0, lambda zs: fun(*zs)).y[:, -1]
    val1_desired = np.array(p(*z1))

    assert pytest.approx(val1_actual, abs=0.01) == val1_desired


def test_int(client):
    z0 = np.array([0.01, 0.01, 1.0])
    z1 = np.array([1.0, 2.0, 3.0])

    client.execute_string('Pf=bload("tests/test_asir/iori-int.bin");')

    def fun(y, mu, sig):
        client.execute_string(f"subst(Pf, y, {y}, mu, {mu}, sig, {sig});")
        return np.array(client.pop_cmo())

    # val0 = v_phis(phi0, *z0)
    val0 = [ -2.26946961e-04
           , -1.82199367e-04
           , 3.10209292e-03
           , -8.15620917e-05
           , 1.12648231e-03
           , -4.11481246e-04
           , 2.93652370e-01]

    val1_actual = hgm.solve(z0, z1, val0, lambda zs: fun(*zs)).y[:, -1]
    # val1_desired = v_phis(phi0, *z1)
    val1_desired = [ -0.005626
                   , -0.00073575
                   , 0.00229936
                   , -0.11128256
                   , 0.03087006
                   , -0.00970615
                   , 0.32661882]

    assert pytest.approx(val1_actual, abs=0.01) == val1_desired


def _test_phi_internal(client, z0, z1, val0, val1_desired):
    def fun(y, mu, sig):
        client.execute_string(f"subst(Pf, y, {y}, mu, {mu}, sig, {sig});")
        return np.array(client.pop_cmo())

    val0 = np.array(val0)
    val1_actual = hgm.solve(z0, z1, val0, lambda zs: fun(*zs)).y[:, -1]
    val1_desired = np.array(val1_desired)

    assert pytest.approx(val1_actual, abs=0.01) == val1_desired
        

def test_phi0(client):
    z0 = np.array([0.01, 0.01, 1.0])
    z1 = np.array([1.0, 2.0, 3.0])

    client.execute_string('Pf=bload("asir-src/pf0-iori2020.bin");')

    # val0 = v_phis(*z0)
    val0 = [ -2.26946961e-04
           , -1.82199367e-04
           , 3.10209292e-03
           , -8.15620917e-05
           , 1.12648231e-03
           , -4.11481246e-04
           , 2.93652370e-01]

    # val1_desired = v_phis(*z1)
    val1_desired = [ -0.005626
                   , -0.00073575
                   , 0.00229936
                   , -0.11128256
                   , 0.03087006
                   , -0.00970615
                   , 0.32661882]

    _test_phi_internal(client, z0, z1, val0, val1_desired)


def test_phi1(client):
    z0 = np.array([0.01, 0.01, 1.0])
    z1 = np.array([1.0, 2.0, 3.0])

    client.execute_string('Pf=bload("asir-src/pf1-iori2020.bin");')

    # val0 = v_phis(phi1, *z0)
    val0 = [ 4.06999348e-02
           , 1.17295408e-02
           , -1.52413311e-04
           , 2.32589161e-01
           , 2.33243840e-01
           , 5.24385018e-04
           , 4.65850780e-03]

    # val1_desired = v_phis(phi1, *z1)
    val1_desired = [ -1.33162665e-02
                   , -7.68652489e-03
                   , -5.89238658e-05
                   , -2.02078366e-01
                   , 2.39832253e-01
                   , 6.48069033e-03
                   , 6.35265839e-01]

    _test_phi_internal(client, z0, z1, val0, val1_desired)


def test_phi2(client):
    z0 = np.array([0.01, 0.01, 1.0])
    z1 = np.array([1.0, 2.0, 3.0])

    client.execute_string('Pf=bload("asir-src/pf2-iori2020.bin");')

    # val0 = v_phis(phi2, *z0)
    val0 = [ -0.00082662
           , 0.00072744
           , 0.01242631
           , 0.00168018
           , 0.00774273
           , 0.21064475
           , 0.47818713]

    # val1_desired = v_phis(phi2, *z1)
    val1_desired = [ -0.14588569
                   , 0.00282539
                   , -0.00425967
                   , -0.83661729
                   , 0.93925337
                   , 0.17417929
                   , 1.89181311]

    _test_phi_internal(client, z0, z1, val0, val1_desired)


def test_phi0_singular_locus(client):
    # mu=0 is singular locus
    z0 = np.array([0.01, -0.01, 1.0])
    z1 = np.array([ 1, 0.001, 1])

    client.execute_string('Pf=bload("asir-src/pf0-iori2020.bin");')

    val0 = v_phis(phi0, *z0)
    val0 = [ -2.26946961e-04
           , -1.82199367e-04
           , 3.10209292e-03
           , -8.15620917e-05
           , 1.12648231e-03
           , -4.11481246e-04
           , 2.93652370e-01]

    val1_desired = v_phis(phi0, *z1)
    val1_desired=[ -0.00063284
                 , -0.00710085
                 , -0.00049771
                 , -0.12877946
                 , 0.04102894
                 , -0.00025141
                 , 0.11023361]

    _test_phi_internal(client, z0, z1, val0, val1_desired)
