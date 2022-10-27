import numpy as np
from models.ou_iori import Model, realize, hgm, naive
from models.ou_iori.phi import v_phis
from models.ou_iori import pfaffian_gamma1on20_sigma1_varob5 as pf
from matplotlib import pyplot as plt

gamma = 1/20
sigma = 1
var_ob = 5
model = Model(gamma, sigma, var_ob)

x0 = 10.0
ts, xs, ys, y_steps = realize(x0, 5000, model=model)

def fun_z0_vphis(z1):
    return [0.1, 0.1, 1], v_phis(0.1, 0.1, 0.1, model)

mu0 = 10.0
sig0 = 1.0
result_hgm = hgm.estimate(mu0, sig0, ys,
                          fun_z0_vphis=fun_z0_vphis,
                          pfs_phi0=pf.phi0,
                          pfs_phi1=pf.phi1,
                          pfs_phi2=pf.phi2)

result_naive = naive.estimate(mu0, sig0, ys, model)


def plot_realization(ax: plt.Axes):
    ax.plot(ts, xs)
    ax.plot(ts[y_steps], ys)
    ax.plot(ts[y_steps], result_naive[0])


fig = plt.figure()
ax = fig.add_subplot()

plot_realization(ax)

plt.show()
