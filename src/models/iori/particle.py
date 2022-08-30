import numpy as np
from scipy import stats
import random


def estimate(ys, xs):
    N = len(xs)
    mus = []
    sigs = []
    for y in ys:
        xxs = [4/5*x + np.random.normal(loc=0, scale=1) for x in xs]
        likelihoods = [float(stats.norm.pdf(y, loc=2*xx/(1 + xx**2), scale=1)) for xx in xxs]
        xs = random.choices(xxs, weights=likelihoods, k=N)
        mus.append(np.mean(xs))
        sigs.append(np.var(xs))

    return mus, sigs
