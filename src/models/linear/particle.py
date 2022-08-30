import numpy as np
import random
from scipy import stats
from . import Model


def transit(x, model: Model):
    v = np.random.normal(loc=0, scale=np.sqrt(model.var_st))
    return model.k * x + v


def likelihood(y, x, model: Model):
    return float(stats.norm.pdf(y, loc=model.l*x, scale=np.sqrt(model.var_ob)))


def estimate(xs0, ys, *, model: Model):

    xs = xs0
    mus = []
    ss = []
    xss = []
    for i, y in enumerate(ys):
        xxs = [transit(x, model) for x in xs]
        ls = [likelihood(y, xx, model) for xx in xxs]
        xs = random.choices(xxs, weights=ls, k=len(xs))

        xss.extend([[i, x] for x in xs])
        mus.append(np.mean(xs))
        ss.append(np.var(xs))


    return mus, ss, np.array(xss)
