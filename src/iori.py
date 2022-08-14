from setup import client
from hgm_estimation import hgm
from hgm_estimation.utils import derivative1, derivative2
import numpy as np
from scipy import stats, integrate
from multiprocessing import Pool
from tqdm import tqdm
from matplotlib import pyplot as plt
from iori_initial import v_phis_cache



def pfs_phi(client, i, zs):
    y, mu, sig = zs
    client.execute_string(f'subst(Pf{i}, y, {y}, mu, {mu}, sig, {sig});')
    return np.array(client.pop_cmo())


for i in range(0, 3):
     client.execute_string(f'Pf{i} = bload("asir-src/pf{i}-iori2020.bin");')


def realize(x0: float, n: int):
    x = x0
    xs = []
    ys = []
    for _ in range(n):
        # state: xx = 4/5*x + noize
        v = np.random.normal(0, 1)
        x = 4/5*x + v

        # observe: y = 2*x/(1+x^2) + noize
        w = np.random.normal(0, 1)
        y = 2*x/(1+x**2) + w

        xs.append(x)
        ys.append(y)

    return xs, ys


def estimate(ys):
    # y = z0[0]
    mu = 0.01
    sig = 1

    mus = []
    sigs = []
    pbar = tqdm(ys)
    for y in pbar:

        z1 = np.array([y, mu, sig])
        
        y0 = 0.01 if y > 0 else -0.01
        mu0 = 0.01 if mu > 0 else -0.01
        z0 = np.array([y0, mu0, 1.0])

        # tqdm.write(f'{z0} -> {z1}')

        v_phi00, v_phi10, v_phi20 = v_phis_cache(*z0)
        r0 = hgm.solve(z0, z1, v_phi00, lambda zs: pfs_phi(client, 0, zs))
        r1 = hgm.solve(z0, z1, v_phi10, lambda zs: pfs_phi(client, 1, zs))
        r2 = hgm.solve(z0, z1, v_phi20, lambda zs: pfs_phi(client, 2, zs))

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


def phi_wrapper(fun, y, mu, sig):
    return fun(y, mu, sig)


def estimate_naive(ys, z0):

    # y = z0[0]
    mu = z0[1]
    sig = z0[2]

    mus = []
    sigs = []
    for y in tqdm(ys):
        with Pool(processes=3) as p:
            args = [(phi0, y, mu, sig)
                   ,(phi1, y, mu, sig)
                   ,(phi2, y, mu, sig)
                   ]
            p0, p1, p2 = p.starmap(phi_wrapper, args)
        tqdm.write(f'p0 = {p0}')
        tqdm.write(f'p1 = {p1}')
        tqdm.write(f'p2 = {p2}')

        mu = p1 / p0
        sig = p2 / p0 - mu**2

        mus.append(mu)
        sigs.append(sig)

    return mus, sigs


def p_mul(x, xp, y, mu, sig):
    a = stats.norm.pdf(x, loc=4/5*xp, scale=1)
    b = stats.norm.pdf(y, loc=2*x/(1+x**2), scale=1)
    c = stats.norm.pdf(xp, loc=mu, scale=np.sqrt(sig))
    return a*b*c


def dblquad_inf(fun, args):
    return integrate.dblquad(
            fun,
            -np.inf, np.inf,
            lambda _: -np.inf, lambda _: np.inf,
            args = args
           )[0]


def phi0(y, mu, sig):
    return dblquad_inf(p_mul, (y, mu, sig))


def phi1(y, mu, sig):
    return dblquad_inf(lambda x, xp, y, mu, sig: x * p_mul(x, xp, y, mu, sig), (y, mu, sig))


def phi2(y, mu, sig):
    return dblquad_inf(lambda x, xp, y, mu, sig: x * x * p_mul(x, xp, y, mu, sig), (y, mu, sig))


DERIV_ORD = [ [1, 0, 1]
            , [0, 1, 1]
            , [0, 0, 2]
            , [1, 0, 0]
            , [0, 1, 0]
            , [0, 0, 1]
            , [0, 0, 0]
            ]

# [dsig*dy,dsig*dmu,dsig^2,dy,dmu,dsig,1]
def v_phi(fun, y, mu, sig, ord):
    print(f'[DEBUG] {fun}({y}, {mu}, {sig}, {ord})')
    match ord:
        case [1, 0, 1]:
            return derivative2(lambda  y, sig: fun(y, mu, sig),  [y, sig], [1, 1])
        case [0, 1, 1]:
            return derivative2(lambda mu, sig: fun(y, mu, sig), [mu, sig], [1, 1])
        case [0, 0, 2]:
            return derivative1(lambda     sig: fun(y, mu, sig),       sig, 2)
        case [1, 0, 0]:
            return derivative1(lambda       y: fun(y, mu, sig),         y, 1)
        case [0, 1, 0]:
            return derivative1(lambda      mu: fun(y, mu, sig),        mu, 1)
        case [0, 0, 1]:
            return derivative1(lambda     sig: fun(y, mu, sig),       sig, 1)
        case [0, 0, 0]:
            return fun(y, mu, sig)
        case _:
            raise NotImplementedError()


def v_phis(y, mu, sig):
    args0 = [(phi0, y, mu, sig, ord) for ord in DERIV_ORD]
    args1 = [(phi1, y, mu, sig, ord) for ord in DERIV_ORD]
    args2 = [(phi2, y, mu, sig, ord) for ord in DERIV_ORD]
    args = [*args0, *args1, *args2]

    with Pool(processes=12) as p:
        r = p.starmap(v_phi, args)
        r1 = r[:7]
        r2 = r[7:14]
        r3 = r[14:]

    return r1, r2, r3


y0 = 0.01
mu0 = 0.01
sig0 = 1.0
# v_phi00, v_phi10, v_phi20 = v_phis(y0, mu0, sig0)
v_phi00, v_phi10, v_phi20 = v_phis_cache(y0, mu0, sig0)

xs, ys = realize(10, 100)
result = estimate(ys)

# result_naive = estimate_naive(ys, np.array([y0, mu0, sig0]))

client.send_shutdown()

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(range(len(xs)), xs)
ax.plot(range(len(result[0])), result[0])
# ax.plot(range(len(result_naive[0])), result_naive[0])
fig.show()
plt.show()
