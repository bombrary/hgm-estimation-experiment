import numpy as np
from models.iori import realize
from models.iori import particle, hgm
from tqdm import tqdm
from timeit import timeit

x0 = 10
y0 = 0.01
mu0 = 10.0
sig0 = 1.0


def times_hgm(N):
    times_hgm = []
    for _ in tqdm(range(N)):
        xs, ys = realize(x0, 100)
        time = timeit(lambda: hgm.estimate(mu0, sig0, ys, log=False), number=1)
        times_hgm.append(time)

    return times_hgm

# print(np.mean(timeit_hgm(10)))
# 50 times mean: 0.5659025116999328
time_hgm = 0.5659025116999328

# (Not Using Cython) 50 times mean: 2.1645398911600022
# (Not Using Cython) time_hgm = 2.1645398911600022


def particle_estimate(ys, size):
    xxs = np.random.normal(loc=mu0, scale=np.sqrt(sig0), size=size)
    particle.estimate(ys, xxs)


def time_particle(time_hgm, l0, r0):
    l = l0
    r = r0
    while l < r:
        mid = (l + r) // 2
        times_par = []
        for _ in tqdm(range(50)):
            xs, ys = realize(x0, 100)
            time = timeit(lambda: particle_estimate(ys, mid), number=1)
            times_par.append(time)

        time_ave = np.mean(times_par)
        print(f'{mid}: {time_ave}')

        if time_ave <= time_hgm:
            l = mid
        else:
            r = mid

print(time_particle(time_hgm, 1, 100))
## particle = 77
