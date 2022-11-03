import numpy as np
from scipy import stats
import random
from . import Model
from tqdm import tqdm

def estimate(ys, xs, model: Model, *, disable_tqdm=False):
    N = len(xs)
    mus = []
    sigs = []
    for y in tqdm(ys, disable=disable_tqdm):
        xxs = [model.k*x + np.random.normal(loc=0, scale=np.sqrt(model.var_st)) for x in xs]
        likelihoods = [float(stats.norm.pdf(y, loc=2*xx/(1 + xx**2), scale=np.sqrt(model.var_ob))) for xx in xxs]
        xs = random.choices(xxs, weights=likelihoods, k=N)
        mus.append(np.mean(xs))
        sigs.append(np.var(xs))

    return mus, sigs