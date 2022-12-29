import numpy as np
from scipy import stats
import random
from . import Model
from . import dx as dx_OU
from tqdm import tqdm

def time_evolution(x0, dt, model: Model):
    x = x0
    for _ in np.arange(0, 1, dt):
        x += dx_OU(x, dt, gamma=model.gamma, sigma=model.sigma)
    return x

def time_evolution_disc(x0, model: Model):
    return model.k*x0 + np.random.normal(loc=0, scale=np.sqrt(model.var_st))

def estimate(ys, xs, model: Model, *, disable_tqdm=False, dt=0.01):
    N = len(xs)
    mus = []
    sigs = []
    for y in tqdm(ys, disable=disable_tqdm):
        xxs = [time_evolution(x, dt, model) for x in xs]
        likelihoods = [float(stats.norm.pdf(y, loc=2*xx/(1 + xx**2), scale=np.sqrt(model.var_ob))) for xx in xxs]
        xs = random.choices(xxs, weights=likelihoods, k=N)
        mus.append(np.mean(xs))
        sigs.append(np.var(xs))

    return mus, sigs
