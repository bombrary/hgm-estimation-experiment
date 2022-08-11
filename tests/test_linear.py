import pytest
from hgm_estimation import hgm
from src.models.linear.phi import v_phis_analytic
from src.models.linear import Model
from src.utils import pfs_phi
import numpy as np


def gen_pfaffian(client, model: Model):
    k, var_st, l, var_ob = model.to_frac()
    client.load("asir-src/phi-linear.rr")
    client.execute_string(f'Pfs = gen_pfaffian({k}, {var_st}, {l}, {var_ob});')


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
