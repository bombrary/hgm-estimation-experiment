import numpy as np
from .models.ou_iori import Model, realize, hgm, naive, phi
from .models.ou_iori.phi import v_phi
from .models.ou_iori import pfaffian_gamma1on20_sigma1_varob5 as pf
from matplotlib import pyplot as plt


cache = {'phi0': {}, 'phi1': {}, 'phi2': {}}
def v_phis0_cache(phi, z0):
    key = str(z0)
    if key in cache[phi.__name__]:
        return cache[phi.__name__][key]
    else:
        res = v_phi(phi, z0[0], z0[1], z0[2], model) 
        cache[phi.__name__][key] = res
        return res


def fact_vphi1(y, mu, sig):
    c0 = 94302
    c1 = 99179
    c2 = 49569
    return (c0*sig+c1)*y + c2*mu 

    C0 = 94302;
def fun_z0_vphi1(z1):
    [y1, mu1, sig1] = z1
    C = fact_vphi1(y1, mu1, sig1)
    if mu1 > 0 and C > 0:
        y0, mu0, sig0 = 0.01, 0.01, 0.01
    elif mu1 < 0 and C < 0:
        y0, mu0, sig0 = -0.01, -0.01, 0.01
    elif mu1 < 0 and C > 0:
        y0, mu0, sig0 = 0.01, -0.01, 0.01
    else:
        y0, mu0, sig0 = -0.01, 0.01, 0.01

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
    ax.plot(ts, xs)
    ax.plot(ts[y_steps], ys, label="observation")
    ax.plot(ts[y_steps], result_naive[0], label='naive')
    ax.plot(ts[y_steps], result_hgm[0], label='hgm')
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.legend()


if __name__ == '__main__':
    gamma = 1/20
    sigma = 1
    var_ob = 1
    model = Model(gamma, sigma, var_ob)

    x0 = 10.0
    ts, xs, ys, y_steps = realize(x0, 5000, model=model)
    print(f'ys: {ys}')

    mu0 = 10.0
    sig0 = 1.0
    result_hgm = hgm.estimate(mu0, sig0, ys,
                              fun_z0_vphi0=lambda z1: fun_z0_vphi02(phi.phi0, z1),
                              fun_z0_vphi1=lambda z1: fun_z0_vphi1(z1),
                              fun_z0_vphi2=lambda z1: fun_z0_vphi02(phi.phi2, z1),
                              pfs_phi0=pf.phi0,
                              pfs_phi1=pf.phi1,
                              pfs_phi2=pf.phi2)

    result_naive = naive.estimate(mu0, sig0, ys, model)

    fig = plt.figure()
    ax = fig.add_subplot()

    plot_realization(ax)

    plt.show()
