import numpy as np
from matplotlib import pyplot as plt
from models.iori import realize
from models.iori import particle, ekf, ukf, naive, hgm


x0 = 10
xs, ys = realize(x0, 100)
y0 = 0.01
mu0 = 10.0
sig0 = 1.0

mus, sigs = hgm.estimate(mu0, sig0, ys, log=True, atol=1e-10, rtol=1e-6)
# mus_naive, sigs_naive = naive.estimate(mu0, sig0, ys) mus_particle, sigs_particle = particle.estimate(ys, np.random.normal(loc=mu0, scale=np.sqrt(sig0), size=100))
xxs = np.random.normal(loc=mu0, scale=np.sqrt(sig0), size=70)
mus_particle, sigs_particle = particle.estimate(ys, xxs)
mus_ekf, sigs_ekf = ekf.estimate(mu0, sig0, ys, k=4/5, var_st=1, var_ob=1)
mus_ukf, sigs_ukf = ukf.estimate(mu0, sig0, ys, 0.5)


fig: plt.Figure  = plt.figure()
ax = fig.add_subplot()


def plot_state(ax, xs, mus, mus_particle, mus_ekf, mus_ukf):
    N = len(mus)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.set_title("estimation result")
    ax.plot(range(N), xs          , label="realized state" )
    ax.plot(range(N), mus         , label="HGM")
    # ax.plot(range(N), mus_naive   , label="naive")
    ax.plot(range(N), mus_particle, label="particle")
    ax.plot(range(N), mus_ekf     , label="EKF")
    ax.plot(range(N), mus_ukf     , label="UKF")
    ax.legend()

def boxplot_diff(ax, xs, mus, mus_particle, mus_ekf, mus_ukf):
    xs = np.array(xs, dtype=np.float64)
    mus = np.array(mus, dtype=np.float64)
    # mus_naive = np.array(mus_naive, dtype=np.float64)
    mus_particle = np.array(mus_particle, dtype=np.float64)
    mus_ekf = np.array(mus_ekf, dtype=np.float64)
    mus_ukf = np.array(mus_ukf, dtype=np.float64)

    ds = np.abs(mus - xs)
    # ds_naive = np.abs(mus_naive - xs)
    ds_particle = np.abs(mus_particle - xs)
    ds_ekf = np.abs(mus_ekf - xs)
    ds_ukf = np.abs(mus_ukf - xs)

    ax.set_ylabel(r"$|\mathrm{estimate} - \mathrm{state}|$")

    # ax.boxplot([ds, ds_particle, ds_ekf, ds_ukf],
    #            vert=True,
    #            labels=['HGM', 'particle', 'EKF', 'ukf'])
    ax.boxplot([ds, ds_particle],
               vert=True,
               labels=['HGM', 'particle'])
    ax.set_title("precision of estimated state")


def scatter_diff(ax, xs, mus, mus_particle):
    xs = np.array(xs, dtype=np.float64)
    mus = np.array(mus, dtype=np.float64)
    mus_particle = np.array(mus_particle, dtype=np.float64)

    ds = np.abs(mus - xs)
    ds_particle = np.abs(mus_particle - xs)

    N = len(ds)
    ax.scatter([0] * N, ds)
    ax.scatter([1] * N, ds_particle)

# plot_state(ax,[x0, *xs]
#              ,[mu0, *mus         ]
#              ,[mu0, *mus_particle]
#              ,[mu0, *mus_ekf     ]
#              ,[mu0, *mus_ukf     ])

scatter_diff(ax, xs, mus, mus_particle)

plt.show()

