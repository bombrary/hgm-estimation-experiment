import numpy as np
from tqdm import tqdm
from . import Model

def sigma_point(mu, var, kappa):
    n = 1
    kais = [mu,
            mu + np.sqrt((n+kappa)*var),
            mu - np.sqrt((n+kappa)*var)]

    return np.array(kais)

def sigma_weight(kappa):
    n = 1
    ws = [kappa/(n+kappa),
        1/(n+kappa)/2,
        1/(n+kappa)/2]
    return np.array(ws)


@np.vectorize
def f(x, model: Model):
    return model.k*x


@np.vectorize
def h(x):
    return 2*x/(1+x**2)


def mu_sig(xs, weights):
    mu = sum(weights * xs)
    sig = sum(weights * xs**2) - mu**2
    return mu, sig


def estimate(mu0, sig0, ys, kappa, model: Model):
    ws = sigma_weight(kappa)

    mu = mu0
    sig = sig0

    mus = []
    sigs = []
    for y in ys:
        kais = sigma_point(mu, sig, kappa)
        kais_pred = f(kais, model)
        mu_pred, sig_pred = mu_sig(kais_pred, ws)
        sig_pred = sig_pred + model.var_st

        kais = sigma_point(mu_pred, sig_pred, kappa)
        kais_obs = h(kais)
        mu_obs, sig_obs = mu_sig(kais_obs, ws)


        cov = sum(ws*(kais_pred-mu_pred)*(kais_obs-mu_obs))
        gain = cov / (sig_obs + model.var_ob)

        mu = mu_pred + gain*(y - mu_obs)
        sig = sig_pred - gain**2*sig_obs

        mus.append(mu)
        sigs.append(sig)

    return mus, sigs
