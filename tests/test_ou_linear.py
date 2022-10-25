import numpy as np
from src.models.ou_linear import pfaffian_1on20_1_1_5 as pf
from src.models.ou_linear import Model
from src.models.linear.phi import phi0_analytic
from hgm_estimation import hgm


def test_phi0():
    model = Model(1/20, 1, 1, 5)
    linear_model = model.to_linear_model()
    z0 = np.array([1, 0, 1])
    z1 = np.array([3, 2, 5])

    v_phi00 = phi0_analytic(z0[0], z0[1], 1/z0[2], model=linear_model)
    result = hgm.solve(z0, z1, [v_phi00], pf.phi0)
    actual = result.y[:, -1][0]
    desired = phi0_analytic(z1[0], z1[1], 1/z1[2], model=linear_model)
    assert actual == desired
