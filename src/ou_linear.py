from models.linear import phi
from models.linear import kalman
from models import ou_linear
from hgm_estimation import estimation
from utils import pfs_phi
from matplotlib import pyplot as plt
from setup import client


ou_model = ou_linear.Model(1/20, 1, 1, 5)
model = ou_model.to_linear_model()
k, var_st, l, var_ob = model.to_frac()


client.load("asir-src/phi-linear.rr")
client.execute_string(f'Pfs = gen_pfaffian({k}, {var_st}, {l}, {var_ob});')


y0 = 0.1
mu0 = 0
lam0 = 1.0
x0 = 10
ts, xs, ys, y_steps = ou_linear.realize(x0, 5000, model=ou_model)
v_phi00, v_phi10, v_phi20 = phi.v_phis_analytic(y0, mu0, lam0, model=model)


result = estimation.run( y0, mu0, lam0, ys
                       , v_phi00 = v_phi00
                       , v_phi10 = v_phi10
                       , v_phi20 = v_phi20
                       , pfs_phi0 = lambda zs: pfs_phi(client, 0, zs)
                       , pfs_phi1 = lambda zs: pfs_phi(client, 1, zs)
                       , pfs_phi2 = lambda zs: pfs_phi(client, 2, zs)
                       )

result_kalman = kalman.estimate(0.0, 1.0, ys, model=model)

# %%

fig: plt.Figure = plt.figure()
ax: plt.Axes = fig.add_subplot()
ax.cla()

ax.set_xscale('linear')
n = len(xs)
ax.plot(ts, xs, label="state")
ax.plot(ts[y_steps], ys, label="observe")
ax.plot(ts[y_steps], result_kalman[0], label="estimate-kalman")
ax.plot(ts[y_steps], result.mus, label="estimate-hgm")
ax.set_xlabel("time")
ax.set_title(r"$\gamma=1/20$, $\sigma=1$, var of state noise=10")

ax.legend()
plt.show()

client.send_shutdown()
