from hgm_estimation import hgm
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from models.iori import realize
from models.iori.iori_initial import v_phis_cache
from models.iori import particle, ekf, ukf, naive
from models.iori import pfaffian



def create_z0(y1, mu1):
    y0 = 0.01 if y1 > 0 else -0.01
    mu0 = 0.01 if mu1 > 0 else -0.01
    return np.array([y0, mu0, 1.0])


def estimate(mu0, sig0, ys):
    # y = z0[0]
    mu = mu0
    sig = sig0

    mus = []
    sigs = []
    pbar = tqdm(ys)
    for y in pbar:

        z1 = np.array([y, mu, sig])
        z0 = create_z0(z1[0], z1[1])

        # tqdm.write(f'{z0} -> {z1}')

        v_phi00, v_phi10, v_phi20 = v_phis_cache(*z0)
        r0 = hgm.solve(z0, z1, v_phi00, lambda zs: pfaffian.phi0(zs))
        r1 = hgm.solve(z0, z1, v_phi10, lambda zs: pfaffian.phi1(zs))
        r2 = hgm.solve(z0, z1, v_phi20, lambda zs: pfaffian.phi2(zs))

        v_phi0 = r0.y[:, -1]
        v_phi1 = r1.y[:, -1]
        v_phi2 = r2.y[:, -1]

        # tqdm.write(f'v_phi0 = {list(v_phi0)}')
        # tqdm.write(f'v_phi1 = {list(v_phi1)}')
        # tqdm.write(f'v_phi2 = {list(v_phi2)}')

        mu = v_phi1[-1] / v_phi0[-1]
        sig = v_phi2[-1] / v_phi0[-1] - mu**2

        mus.append(mu)
        sigs.append(sig)

    return mus, sigs




x0 = 10
xs, ys = realize(x0, 100)
y0 = 0.01
mu0 = 10.0
sig0 = 1.0

mus, sigs = estimate(mu0, sig0, ys)
# mus_naive, sigs_naive = naive.estimate(mu0, sig0, ys)
mus_particle, sigs_particle = particle.estimate(ys, np.random.normal(loc=mu0, scale=np.sqrt(sig0), size=100))
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

def scatter_diff(ax, xs, mus, mus_particle, mus_ekf, mus_ukf):
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

    ax.boxplot([ds, ds_particle, ds_ekf, ds_ukf],
               vert=True,
               labels=['HGM', 'particle', 'EKF', 'ukf'])
    ax.set_title("precision of estimated state")

# plot_state(ax,[x0, *xs]
#              ,[mu0, *mus         ]
#              ,[mu0, *mus_particle]
#              ,[mu0, *mus_ekf     ]
#              ,[mu0, *mus_ukf     ])

scatter_diff(ax, xs, mus, mus_particle, mus_ekf, mus_ukf)

plt.show()
