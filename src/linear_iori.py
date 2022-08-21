# from setup import client
from hgm_estimation import hgm
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from matplotlib import pyplot as plt
from models.linear_iori.phi import phi0, phi1, phi2, v_phis
from models.linear_iori import Model
from functools import lru_cache

def pfs_phi(client, i, zs):
    y, mu, sig = zs
    client.execute_string(f'subst(Pfs[{i}], y, {y}, mu, {mu}, sig, {sig});')
    return np.array(client.pop_cmo())


@lru_cache()
def v_phis_cache(y, mu, sig, model: Model):
    match (y, mu, sig):
        case (0.01, 0.01, 1.0):
            v_phi0 = [-0.0005772025912786916, -0.00032914851400001055, 0.024100318241071506, 0.000468158879163294, 0.0015742398543350244, -0.008887149807651484, 0.2960716698516484]
            v_phi1 = [0.08339103913456401, 0.02594002037519319, -0.000554957907752951, 0.18876894532300842, 0.2695322423569664, 0.0010935441532265722, 0.004583042557587371]
            v_phi2 = [-5.440263517453303e-05, 0.0022555725898154577, 0.046261847908013465, 0.0028323778049066384, 0.009453137159276537, 0.2907567646585063, 0.29048295203032154]
        case (-0.01, 0.01, 1.0):
            v_phi0 = [-0.0010448823056785628, 0.001292914914019505, 0.024080516469737745, 0.0030353945530414705, -0.0019294649645784645, -0.008870927851289956, 0.2960366317165652]
            v_phi1 = [0.08340320567867626, 0.025960613027911857, 0.0004745857618568619, 0.18878389504862206, 0.26949887946365564, -0.0005745324348027203, 0.0008072338867572403]
            v_phi2 = [0.002714412319915205, -0.0004045556112330928, 0.046283254617751624, 0.00444353988121593, 0.002176478875348664, 0.290730161942504, 0.2904101872298824]
        case (0.01, -0.01, 1.0):
            v_phi0 = [0.0010448823056785628, -0.001292914914019505, 0.024080516469737745, -0.0030353945530414705, 0.0019294649645784645, -0.008870927851289956, 0.2960366317165652]
            v_phi1 = [0.08340320567870337, 0.02596061302788475, -0.0004745857618568619, 0.18878389504862206, 0.26949887946365564, 0.0005745324348027203, -0.0008072338867572403]
            v_phi2 = [-0.002714412319915205, 0.0004045556112330928, 0.046283254617751624, -0.00444353988121593, -0.002176478875348664, 0.290730161942504, 0.2904101872298824]
        case (-0.01, -0.01, 1.0):
            v_phi0 = [0.0005772025912786916, 0.00032914851400001055, 0.024100318241071506, -0.000468158879163294, -0.0015742398543350244, -0.008887149807651484, 0.2960716698516484]
            v_phi1 = [0.08339103913456401, 0.02594002037519319, 0.000554957907752951, 0.18876894532300842, 0.2695322423569664, -0.0010935441532265722, -0.004583042557587371]
            v_phi2 = [5.440263517453303e-05, -0.0022555725898154577, 0.046261847908013465, -0.0028323778049066384, -0.009453137159276537, 0.2907567646585063, 0.29048295203032154]
        case _:
            v_phi0, v_phi1, v_phi2 = v_phis(y, mu, sig, model)
            tqdm.write(f'v_phi0 = {list(v_phi0)}')
            tqdm.write(f'v_phi1 = {list(v_phi1)}')
            tqdm.write(f'v_phi2 = {list(v_phi2)}')
    return v_phi0, v_phi1, v_phi2


def realize(x0: float, n: int, model: Model):
    x = x0
    xs = []
    ys = []
    for _ in range(n):
        # state: xx = k*x + noize
        v = np.random.normal(0, np.sqrt(model.var_st))
        x = model.k*x + v

        # observe: y = 2*x/(1+x^2) + noize
        w = np.random.normal(0, np.sqrt(model.var_ob))
        y = 2*x/(1+x**2) + w

        xs.append(x)
        ys.append(y)

    return xs, ys


def create_z0(y1, mu1):
    y0 = 0.01 if y1 > 0 else -0.01
    mu0 = 0.01 if mu1 > 0 else -0.01
    return np.array([y0, mu0, 1.0])


def estimate(ys, model):
    # y = z0[0]
    mu = 0.01
    sig = 1

    mus = []
    sigs = []
    pbar = tqdm(ys)
    for y in pbar:

        z1 = np.array([y, mu, sig])
        z0 = create_z0(z1[0], z1[1])

        # tqdm.write(f'{z0} -> {z1}')

        v_phi00, v_phi10, v_phi20 = v_phis_cache(*z0, model=model)
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


def phi_wrapper(fun, y, mu, sig, model):
    return fun(y, mu, sig, model)


def estimate_naive(ys, z0, model):

    # y = z0[0]
    mu = z0[1]
    sig = z0[2]

    mus = []
    sigs = []
    for y in tqdm(ys):
        with Pool(processes=3) as p:
            args = [(phi0, y, mu, sig, model)
                   ,(phi1, y, mu, sig, model)
                   ,(phi2, y, mu, sig, model)
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


model = Model(39/40, 0.1, 1)
# client.load("asir-src/phi-linear-iori.rr")
# client.execute_string("Pfs = gen_pfaffian(39/40, 1/10, 1);")

xs, ys = realize(10, 100, model)
# result = estimate(ys, model)

ys = ys[:50]
result_naive = estimate_naive(ys, np.array([0.01, 0.01, 1.0]), model)

# client.send_shutdown()

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(range(len(xs)), xs)
# ax.plot(range(len(result[0])), result[0])
ax.plot(range(len(result_naive[0])), result_naive[0])
fig.show()
plt.show()
