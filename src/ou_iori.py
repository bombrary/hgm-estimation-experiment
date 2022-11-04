import numpy as np
from .models.ou_iori import Model, realize, hgm, naive, phi, ukf, ekf, particle
from .models.ou_iori.phi import v_phi
from .models.ou_iori import pfaffian_gamma1on20_sigma1_varob1 as pf
from matplotlib import pyplot as plt
import time


cache = {'phi0': {
            '[0.1, 0.1, 0.01]': np.array([-0.00563109, -0.00298776, 0.02750941, 0.00460847, 0.01560604, -0.01132219, 0.29803653]),
            '[-0.1, 0.1, 0.01]': np.array([-0.01111872, 0.01373545, 0.02538121, 0.03071762, -0.01986947, -0.00963564, 0.29447771]),
            },
         'phi1': {
            '[1.0, 1.0, 0.01]': np.array([-0.02734957, -0.04698884, -0.00264924, -0.05588302, 0.31461153, 0.02847173, 0.39722788]),
            '[-1.0, 1.0, 0.01]': np.array([0.02771979, 0.0509592, -0.00099335, 0.13752545, 0.13333263, -0.02231257, 0.04633765]),
            '[-1.0, -1.0, 0.01]': np.array([-0.02734957, -0.04698884, 0.00264924, -0.05588302,  0.31461153, -0.02847173, -0.39722788]),
            '[1.0, -1.0, 0.01]': np.array([0.02771979, 0.0509592, 0.00099335, 0.13752545, 0.13333263, 0.02231257, -0.04633765]),
         },
         'phi2': {
             '[0.1, 0.1, 0.01]': np.array([0.00090885, 0.02222127, 0.0438181, 0.02691856, 0.08860754, 0.27355444, 0.26891495]),
             '[-0.1, 0.1, 0.01]': np.array([0.02535506, -0.00415421, 0.04596719, 0.04082861, 0.02016327, 0.27090254, 0.26208822]),
         }}
def v_phis0_cache(phi, z0):
    key = str(z0)
    if key in cache[phi.__name__]:
        return cache[phi.__name__][key]
    else:
        res = v_phi(phi, z0[0], z0[1], z0[2], model) 
        cache[phi.__name__][key] = res
        return res


# CAUTION: this coefficient is used by ONLY pfaffian_gamma1on20_sigma1_varob1.
def fact_vphi1(y, mu, sig):
    c0 = 94302
    c1 = 99179
    c2 = 49569
    return (c0*sig+c1)*y + c2*mu 


def fun_z0_vphi1(z1):
    [y1, mu1, sig1] = z1
    C = fact_vphi1(y1, mu1, sig1)
    if mu1 > 0 and C > 0:
        y0, mu0, sig0 = 1.0, 1.0, 0.01
    elif mu1 < 0 and C < 0:
        y0, mu0, sig0 = -1.0, -1.0, 0.01
    elif mu1 < 0 and C > 0:
        y0, mu0, sig0 = 1.0, -1.0, 0.01
    else:
        y0, mu0, sig0 = -1.0, 1.0, 0.01

    z0 = [y0, mu0, sig0]
    return z0, v_phis0_cache(phi.phi1, z0)


def fun_z0_vphi02(phi, z1):
    [_, mu1, _] = z1
    if mu1 > 0:
        y0 = 0.1
        mu0 = 0.1
        sig0 = 0.01
    else:
        y0 = 0.1
        mu0 = -0.1
        sig0 = 0.01

    z0 = [mu0, y0, sig0]
    return z0, v_phis0_cache(phi, z0)


def plot_realization(ax: plt.Axes):
    ax.plot(ts, xs, color='gray', label="state")
    # ax.plot(ts[y_steps], ys, label="observation")
    # ax.plot(ts[y_steps], result_naive[0], label='naive')
    # ax.plot(ts[y_steps], result_ukf[0], label='ukf', color='forestgreen')
    ax.plot(ts[y_steps], result_ekf[0], label='ekf', color='darkblue')
    ax.plot(ts[y_steps], result_particle[0], label='particle', color='forestgreen')
    ax.plot(ts[y_steps], result_hgm[0], label='hgm', color='red')
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.legend()


if __name__ == '__main__':
    gamma = 1/20
    sigma = 1
    var_ob = 1
    model = Model(gamma, sigma, var_ob)

    x0 = 10.0

    ts, xs, ys, y_steps = realize(x0, 1000, model=model, dt=0.1)

    mu0 = 10.0
    sig0 = 1.0

    result_hgm = hgm.estimate(mu0, sig0, ys,
                              fun_z0_vphi0=lambda z1: fun_z0_vphi02(phi.phi0, z1),
                              fun_z0_vphi1=lambda z1: fun_z0_vphi1(z1),
                              fun_z0_vphi2=lambda z1: fun_z0_vphi02(phi.phi2, z1),
                              pfs_phi0=pf.phi0,
                              pfs_phi1=pf.phi1,
                              pfs_phi2=pf.phi2)

    xxs = np.random.normal(loc=mu0, scale=np.sqrt(sig0), size=100)
    result_particle = particle.estimate(ys, xxs, model)

    # result_naive = naive.estimate(mu0, sig0, ys, model)
    # result_ukf = ukf.estimate(mu0, sig0, ys, 0.1, model)
    result_ekf = ekf.estimate(mu0, sig0, ys, model)


    fig = plt.figure()
    ax = fig.add_subplot()

    plot_realization(ax)

    plt.show()
