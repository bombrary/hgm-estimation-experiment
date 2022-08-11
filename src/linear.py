from models import linear
from models.linear import phi
from models.linear import kalman
from hgm_estimation import estimation
from utils import pfs_phi
from matplotlib import pyplot as plt
from setup import client


model = linear.Model(1, 1/10, 1, 10)
k, var_st, l, var_ob = model.to_frac()


client.load("asir-src/phi-linear.rr")
client.execute_string(f'Pfs = gen_pfaffian({k}, {var_st}, {l}, {var_ob});')


y0 = 0.1
mu0 = 0.1
lam0 = 1.0
x0 = 10.0
xs, ys = linear.realize(x0, 50, model=model)
v_phi00, v_phi10, v_phi20 = phi.v_phis_analytic(y0, mu0, lam0, model=model)
result = estimation.run( y0, mu0, lam0, ys
                       , v_phi00 = v_phi00
                       , v_phi10 = v_phi10
                       , v_phi20 = v_phi20
                       , pfs_phi0 = lambda zs: pfs_phi(client, 0, zs)
                       , pfs_phi1 = lambda zs: pfs_phi(client, 1, zs)
                       , pfs_phi2 = lambda zs: pfs_phi(client, 2, zs)
                       )


result_kalman = kalman.estimate(1.0, 10.0, ys, model=model)


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
plt.show()

client.send_shutdown()
