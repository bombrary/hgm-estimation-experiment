from hgm_estimation.utils import derivative1, derivative2
from hgm_estimation import hgm
from scipy import integrate, stats
import numpy as np
from multiprocessing import Pool
import pytest
from src.models.linear_iori.phi import phi0, phi1, p_obs, p_st, p_gauss, p_mul, v_phi2
from src.models.linear_iori import Model
from fractions import Fraction



@pytest.mark.parametrize(('model', 'z0', 'z1'), [
    (Model(1, 1, 1), [0.01, 0.02], [2, 1]),
    (Model(1, 1, 1), [0.01, 0.02], [1, 2]),
    (Model(1, 1, 0.1), [0.01, 0.02], [1, 2]),
    (Model(1, 1, 0.1), [0.01, 0.02], [2, 1]),
    (Model(1, 1, 10), [0.01, 0.02], [1, 2]),
    (Model(1, 1, 10), [0.01, 0.02], [2, 1]),
])
def test_obs(client, model, z0, z1):
    client.load("tests/test_asir/iori-obs.rr");
    client.execute_string(f"Pf = gen_pfaffian({Fraction(model.var_ob)});")

    def fun(x, y):
        client.execute_string(f"matrix_matrix_to_list(subst(Pf, x, {x}, y, {y}));")
        return np.array(client.pop_cmo())

    p = lambda x, y: p_obs(x, y, model)

    val0 = p(*z0)
    val1_actual = hgm.solve(z0, z1, [val0], lambda zs: fun(*zs)).y[:, -1][0]
    val1_desired = p(*z1)

    assert pytest.approx(val1_actual, abs=0.01) == val1_desired


@pytest.mark.parametrize(('model', 'z0', 'z1'), [
    (Model(1, 1, 1), [0.01, 0.02], [1, 2]),
    (Model(1, 1, 1), [0.01, 0.02], [2, 1]),
    (Model(1, 0.1, 1), [0.01, 0.02], [1, 2]),
    (Model(1, 0.1, 1), [0.01, 0.02], [2, 1]),
    (Model(1, 10, 1), [0.01, 0.02], [1, 2]),
    (Model(1, 10, 1), [0.01, 0.02], [2, 1]),
    (Model(4/5, 10, 1), [0.01, 0.02], [1, 2]),
    (Model(4/5, 10, 1), [0.01, 0.02], [2, 1]),
])
def test_st(client, model, z0, z1):
    client.load("tests/test_asir/iori-st.rr");
    client.execute_string(f"Pf = gen_pfaffian({Fraction(model.k)}, {Fraction(model.var_st)});")

    def fun(xx, x):
        client.execute_string(f"matrix_matrix_to_list(subst(Pf, xx, {xx}, x, {x}));")
        return np.array(client.pop_cmo())

    p = lambda xx, x: p_st(xx, x, model)

    val0 = p(*z0)
    val1_actual = hgm.solve(z0, z1, [val0], lambda zs: fun(*zs)).y[:, -1][0]
    val1_desired = p(*z1)

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


@pytest.mark.parametrize(('model', 'z0', 'z1'), [
    (Model(1.0, 1.0, 1.0), [0.01, 0.01, 0.02, 0.01], [1, 2, 3, 4]),
    (Model(0.1, 1.0, 1.0), [0.01, 0.01, 0.02, 0.01], [1, 2, 3, 4]),
    (Model(1, 0.1, 1.0), [0.01, 0.01, 0.02, 0.01], [1, 2, 3, 4]),
    (Model(1, 1, 0.1), [0.01, 0.01, 0.02, 0.01], [1, 2, 3, 4]),
])
def test_mul_gauss_st(client, model, z0, z1):
    client.load("tests/test_asir/iori-mul1.rr");
    client.execute_string(f'Pf = gen_pfaffian({Fraction(model.k)}, {Fraction(model.var_st)}, {Fraction(model.var_ob)});');

    def fun(xx, x, mu, sig):
        client.execute_string(f"subst(Pf, xx, {xx}, x, {x}, mu, {mu}, sig, {sig});")
        return np.array(client.pop_cmo())

    def p(xx, x, mu, sig):
        return p_gauss(x, mu, sig) * p_st(xx, x, model)


    val0 = p(*z0)
    val1_actual = hgm.solve(z0, z1, [val0], lambda zs: fun(*zs)).y[:, -1][0]
    val1_desired = p(*z1)

    assert pytest.approx(val1_actual, abs=0.01) == val1_desired


@pytest.mark.parametrize(('model', 'z0', 'z1'), [
    (Model(1.0, 1.0, 1.0), [0.02, 0.01, 0.01, 0.02, 1.0], [1, 2, 3, 4, 5]),
    (Model(0.1, 1.0, 1.0), [0.02, 0.01, 0.01, 0.02, 1.0], [1, 2, 3, 4, 5]),
    (Model(1.0, 0.1, 1.0), [0.02, 0.01, 0.01, 0.02, 1.0], [1, 2, 3, 4, 5]),
    (Model(1.0, 1.0, 0.1), [0.02, 0.01, 0.01, 0.02, 1.0], [1, 2, 3, 4, 5]),
])
def test_mul(client, model, z0, z1):
    client.load("tests/test_asir/iori-mul2.rr");
    client.execute_string(f'Pf = gen_pfaffian({Fraction(model.k)}, {Fraction(model.var_st)}, {Fraction(model.var_ob)});');

    def fun(xx, x, y, mu, sig):
        client.execute_string(f"subst(Pf, xx, {xx}, x, {x}, y, {y}, mu, {mu}, sig, {sig});")
        return np.array(client.pop_cmo())

    p = lambda xx, x, y, mu, sig: p_mul(xx, x, y, mu, sig, model)

    val0 = p(*z0)
    val1_actual = hgm.solve(z0, z1, [val0], lambda zs: fun(*zs)).y[:, -1][0]
    val1_desired = p(*z1)

    assert pytest.approx(val1_actual, abs=0.01) == val1_desired


@pytest.mark.parametrize(('model', 'z0', 'z1'), [
    (Model(1.0, 1.0, 1.0), [0.01, 0.01, 0.02, 1.0], [1, 2, 3, 4]),
    (Model(0.1, 1.0, 1.0), [0.01, 0.01, 0.02, 1.0], [1, 2, 3, 4]),
    (Model(1.0, 0.1, 1.0), [0.01, 0.01, 0.02, 1.0], [1, 2, 3, 4]),
    (Model(1.0, 1.0, 0.1), [0.01, 0.01, 0.02, 1.0], [1, 2, 3, 4]),
])
def test_int_x(client, model, z0, z1):
    client.load("tests/test_asir/iori-int1.rr");
    client.execute_string(f'Pf = gen_pfaffian({Fraction(model.k)}, {Fraction(model.var_st)}, {Fraction(model.var_ob)});');

    def fun(xx, y, mu, sig):
        client.execute_string(f"subst(Pf, xx, {xx}, y, {y}, mu, {mu}, sig, {sig});")
        return np.array(client.pop_cmo())

    def p(xx, y, mu, sig):
        return integrate.quad(lambda x: p_mul(xx, x, y, mu, sig, model)
                             ,-np.inf
                             ,np.inf
                             )[0]

    val0 = p(*z0)
    val1_actual = hgm.solve(z0, z1, [val0], lambda zs: fun(*zs)).y[:, -1][0]
    val1_desired = p(*z1)

    assert pytest.approx(val1_actual, abs=0.01) == val1_desired


@pytest.mark.parametrize(('model', 'z0', 'z1'), [
    (Model(1.0, 1.0, 1.0), [0.01, 0.01, 0.02, 0.01], [1, 2, 3, 4]),
    (Model(0.1, 1.0, 1.0), [0.01, 0.01, 0.02, 0.01], [1, 2, 3, 4]),
    (Model(1.0, 0.1, 1.0), [0.01, 0.01, 0.02, 0.01], [1, 2, 3, 4]),
    (Model(1.0, 1.0, 0.1), [0.01, 0.01, 0.02, 0.01], [1, 2, 3, 4]),
])
def test_int_xx(client, model, z0, z1):
    client.load("tests/test_asir/iori-int2.rr");
    client.execute_string(f'Pf = gen_pfaffian({Fraction(model.k)}, {Fraction(model.var_st)}, {Fraction(model.var_ob)});');

    def fun(x, y, mu, sig):
        client.execute_string(f"subst(Pf, x, {x}, y, {y}, mu, {mu}, sig, {sig});")
        return np.array(client.pop_cmo())

    def int_pmul(x, y, mu, sig):
        return integrate.quad(lambda xx: p_mul(xx, x, y, mu, sig, model)
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


@pytest.mark.skip(reason="This test spends a long time passsing")
@pytest.mark.parametrize(('model', 'z0', 'z1'), [
    (Model(1, 1, 1), [0.01, 0.01, 1.0], [1.0, 2.0, 3.0]),
    (Model(0.1, 1, 1), [0.01, 0.01, 1.0], [1.0, 2.0, 3.0]),
    (Model(1, 0.1, 1), [0.01, 0.01, 1.0], [1.0, 2.0, 3.0]),
    (Model(1, 1, 0.1), [0.01, 0.01, 1.0], [1.0, 2.0, 3.0]),
])
def test_phi0(client, model, z0, z1):
    client.load("tests/test_asir/iori-phi0.rr")
    client.execute_string(f'Pf=gen_pfaffian({Fraction(model.k)}, {Fraction(model.var_st)}, {Fraction(model.var_ob)});');

    def fun(y, mu, sig):
        client.execute_string(f"subst(Pf, y, {y}, mu, {mu}, sig, {sig});")
        return np.array(client.pop_cmo())

    val0, val1_desired = v_phi2(phi0, z0, z1, model=model)

    val1_actual = hgm.solve(z0, z1, val0, lambda zs: fun(*zs)).y[:, -1]

    assert pytest.approx(val1_actual, abs=0.01) == np.array(val1_desired)


# @pytest.mark.skip(reason="This test spends a long time passsing")
@pytest.mark.parametrize(('model', 'z0', 'z1'), [
      (Model(0.1, 1, 1), [0.01, 0.02, 1.0], [1.0, 2.0, 3.0]),
#     (Model(0.1, 1, 1), [0.01, 0.01, 1.0], [1.0, 2.0, 3.0]),
#     (Model(1, 0.1, 1), [0.01, 0.01, 1.0], [1.0, 2.0, 3.0]),
#     (Model(1, 1, 0.1), [0.01, 0.01, 1.0], [1.0, 2.0, 3.0]),
])
def test_phi1(client, model, z0, z1):
    client.load("tests/test_asir/iori-phi1.rr")
    client.execute_string(f'Pf=gen_pfaffian({Fraction(model.k)}, {Fraction(model.var_st)}, {Fraction(model.var_ob)});');

    def fun(y, mu, sig):
        client.execute_string(f"subst(Pf, y, {y}, mu, {mu}, sig, {sig});")
        return np.array(client.pop_cmo())

    val0, val1_desired = v_phi2(phi1, z0, z1, model=model)

    val1_actual = hgm.solve(z0, z1, val0, lambda zs: fun(*zs)).y[:, -1]

    assert pytest.approx(val1_actual, abs=0.01) == np.array(val1_desired)



# This test SHOULD be fail
@pytest.mark.skip(reason="This test spends a long time passsing")
def test_phi0_singular_locus(client):
    model = Model(4/5, 1, 1)

    # mu=0 is singular locus
    z0 = np.array([0.01, -0.01, 1.0])
    z1 = np.array([ 1, 0.001, 1])

    client.execute_string('Pf=bload("asir-src/pf0-iori2020.bin");')

    val0, val1_desired = v_phi2(phi0, z0, z1, model=model)
    # val0 = [ -2.26946961e-04
    #        , -1.82199367e-04
    #        , 3.10209292e-03
    #        , -8.15620917e-05
    #        , 1.12648231e-03
    #        , -4.11481246e-04
    #        , 2.93652370e-01]
    # val1_desired=[ -0.00063284
    #              , -0.00710085
    #              , -0.00049771
    #              , -0.12877946
    #              , 0.04102894
    #              , -0.00025141
    #              , 0.11023361]

    _test_phi_internal(client, z0, z1, val0, val1_desired)
