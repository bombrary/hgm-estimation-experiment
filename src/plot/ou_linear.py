import numpy as np
from matplotlib import pyplot as plt

ts, xs = np.loadtxt("data/ou_linear_t_states_gamma0.05_sigma1_obsvar10.csv",
            delimiter=",",
            skiprows=1,
            unpack=True)

# t, y, mu, lam, mu_kalman, lam_kalman
ts_y, ys, mus, _, mus_kalman, _ = np.loadtxt("data/ou_linear_obs_mus_gamma0.05_sigma1_obsvar10.csv",
        delimiter=",",
        skiprows=1,
        unpack=True)


fig: plt.Figure = plt.figure()
ax: plt.Axes = fig.add_subplot()
ax.cla()

ax.set_xscale('linear')
n = len(xs)
ax.plot(ts, xs, label="state")
ax.plot(ts_y, ys, label="observe")
ax.plot(ts_y, mus, label="estimate-hgm")
ax.plot(ts_y, mus_kalman, label="estimate-kalman")
ax.set_xlabel("time")
ax.set_title(r"$\gamma=1/20$, $\sigma=1$, var of state noise=10")
fig.set_facecolor('#fafafa')

ax.legend()
plt.show()
