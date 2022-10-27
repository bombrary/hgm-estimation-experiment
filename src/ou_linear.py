from models.linear import phi
from models.linear import kalman
from models import ou_linear
from models.ou_linear import pfaffian_1on20_1_1_5 as pf
from models.ou_linear import hgm
from hgm_estimation import estimation
from matplotlib import pyplot as plt
import numpy as np

ou_model = ou_linear.Model(1/20, 1, 1, 5)
model = ou_model.to_linear_model()
k, var_st, l, var_ob = model.to_frac()


y0 = 0.1
mu0 = 10
sig0 = 1.0
x0 = 10
ts, xs, ys, y_steps = ou_linear.realize(x0, 5000, model=ou_model)

v_phi00, v_phi10, v_phi20 = phi.v_phis_analytic(y0, mu0, sig0, model=model)
def fun_z0_vphis(z0):
    return [y0, mu0, sig0], [v_phi00, v_phi10, v_phi20]

result_hgm = hgm.estimate( mu0, sig0, ys
                         , fun_z0_vphis = fun_z0_vphis
                         , pfs_phi0 = lambda zs: pf.phi0(zs)
                         , pfs_phi1 = lambda zs: pf.phi1(zs)
                         , pfs_phi2 = lambda zs: pf.phi2(zs)
                         )

result_kalman = kalman.estimate(mu0, sig0, ys, model=model)

# %%

fig: plt.Figure = plt.figure()
fig.set_facecolor('#fafafa')

ax = fig.add_subplot()

def plot_estimate(ax: plt.Axes):
    ax.cla()
    ax.set_xscale('linear')
    n = len(xs)
    ax.plot(ts, xs, label="state")
    ax.plot(ts[y_steps], ys, label="observe")
    ax.plot(ts[y_steps], result_kalman[0], label="estimate-kalman")
    ax.plot(ts[y_steps], result_hgm[0], label="estimate-hgm")
    ax.set_xlabel("time")
    ax.set_title(r"$\gamma=1/20$, $\sigma=1$, var of state noise=10")
    ax.legend()

plot_estimate(ax)
plt.show()


# np.savetxt("data/ou_linear_t_states_gamma0.05_sigma1_obsvar10.csv",
#         np.array([ts, xs]).T,
#         delimiter=",",
#         comments="",
#         fmt="%.5f",
#         header="t, x")
# 
# np.savetxt("data/ou_linear_obs_mus_gamma0.05_sigma1_obsvar10.csv",
#         np.array([ts[y_steps], ys, result_hgm[0], result_hgm[1], result_kalman[0], result_kalman[1]]).T,
#         delimiter=",",
#         comments="",
#         fmt="%.5f",
#         header="t, y, mu, sig, mu_kalman, sig_kalman")
