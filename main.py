from hgm_estimation import estimation
import numpy as np
from matplotlib import pyplot as plt

def plot_estimate(ax: plt.Axes, ts, mus, lams):
    sigs = np.sqrt(1 / lams)
    ax.plot(ts, mus, label="estimate")
    ax.fill_between(ts, mus + sigs, mus - sigs, color="lightgray")

# %%

from ox_asir.client import ClientPipe
from hgm_estimation.models import linear
from hgm_estimation.models.linear import phi as linear_phi
from hgm_estimation.models.linear import kalman as linear_kalman
from fractions import Fraction

def pfs_phi(i, zs):
    y = Fraction(zs[0])
    mup = Fraction(zs[1])
    lamp = Fraction(zs[2])
    client.execute_string(f'subst(Pfs[{i}], y, {y}, mup, {mup}, lamp, {lamp});')
    return np.array(client.pop_cmo())

client = ClientPipe(
        openxm_path="/home/bombrary/openxm/OpenXM/bin/openxm",
        args = ["openxm", "ox_asir", "-nomessage"]
        )
# client.send_shutdown()

# %%

model = linear.Model(1, 1/10, 1, 10)
k, var_st, l, var_ob = model.to_frac()

client.load("asir-src/phi-linear.rr")
client.execute_string(f'Pfs = gen_pfaffian({k}, {var_st}, {l}, {var_ob});')

y0 = 0.1
mu0 = 0.1
lam0 = 1.0
x0 = 10.0
xs, ys = linear.realize(x0, 50, model=model)
v_phi00, v_phi10, v_phi20 = linear_phi.v_phis_analytic(y0, mu0, lam0, model=model)
result = estimation.run( y0, mu0, lam0, ys
                       , v_phi00 = v_phi00
                       , v_phi10 = v_phi10
                       , v_phi20 = v_phi20
                       , pfs_phi0 = lambda zs: pfs_phi(0, zs)
                       , pfs_phi1 = lambda zs: pfs_phi(1, zs)
                       , pfs_phi2 = lambda zs: pfs_phi(2, zs)
                       )
result_kalman = linear_kalman.estimate(1.0, 10.0, ys, model=model)

# %%

fig: plt.Figure = plt.figure()
ax: plt.Axes = fig.add_subplot()

ax.set_xscale('linear')
n = len(xs)
ax.plot(range(0, n), xs, label="state")
ax.plot(range(0, n), ys, label="observe")
ax.plot(range(0, len(result.mus)), result.mus, label="estimate-hgm")
ax.plot(range(0, len(result_kalman[0,1:])), result_kalman[0, 1:], label="estimate-kalman")
ax.set_xlabel("time")
ax.set_title("state var=0.1, observe var=10")

ax.legend()
fig.show()

# fig.savefig("/home/bombrary/repos/bombrary/research/seminar/20220704/img/locallevel-stvar0.1-obsvar10.pdf")


# %%

from hgm_estimation.models import ou_linear

ou_model = ou_linear.Model(1/20, 1, 1, 10)
model = ou_model.to_linear_model()
k, var_st, l, var_ob = model.to_frac()

client.load("asir-src/phi-linear.rr")
client.execute_string(f'Pfs = gen_pfaffian({k}, {var_st}, {l}, {var_ob});')

y0 = 0.1
mu0 = 0
lam0 = 1.0
x0 = 10
ts, xs, ys, y_steps = ou_linear.realize(x0, 5000, model=ou_model)
v_phi00, v_phi10, v_phi20 = linear_phi.v_phis_analytic(y0, mu0, lam0, model=model)

result = estimation.run( y0, mu0, lam0, ys
                       , v_phi00 = v_phi00
                       , v_phi10 = v_phi10
                       , v_phi20 = v_phi20
                       , pfs_phi0 = lambda zs: pfs_phi(0, zs)
                       , pfs_phi1 = lambda zs: pfs_phi(1, zs)
                       , pfs_phi2 = lambda zs: pfs_phi(2, zs)
                       )

result_kalman = linear_kalman.estimate(0.0, 1.0, ys, model=model)

# %%

fig: plt.Figure = plt.figure()
ax: plt.Axes = fig.add_subplot()
ax.cla()

ax.set_xscale('linear')
n = len(xs)
ax.plot(ts, xs, label="state")
ax.plot(ts[y_steps], ys, label="observe")
ax.plot(ts[y_steps], result_kalman[0, 1:], label="estimate-kalman")
ax.plot(ts[y_steps], result.mus[1:], label="estimate-hgm")
ax.set_xlabel("time")
ax.set_title(r"$\gamma=1/20$, $\sigma=1$, var of state noise=10")

ax.legend()
fig.show()

# fig.savefig("/home/bombrary/repos/bombrary/research/workreport/2022-06/img/ou-linear-gamma0.05-sigma1.0-stvar10.pdf")

