import numpy as np
from fractions import Fraction

class Model:
    def __init__(self, gamma, sigma, var_ob):
        self.gamma = gamma
        self.sigma = sigma
        self.var_ob = var_ob

        self.k = np.exp(-self.gamma)
        self.var_st = self.sigma**2 * (1 - self.k**2) / (2*self.gamma)


    def to_frac(self):
        return ( Fraction(self.k)
               , Fraction(self.var_st)
               , Fraction(self.var_ob)
               )


def dx(x: float, dt: float, *, gamma: float, sigma: float):
    dW = np.random.normal(0, dt)
    return -gamma * x * dt + sigma * dW


def y(x: float, *, std_ob: float):
    return (2*x)/(1 + x**2) + np.random.normal(0, std_ob)


def realize(x0, iter_max, *, dt = 0.01, model: Model):
    gamma = model.gamma
    sigma = model.sigma
    std_ob = np.sqrt(model.var_ob)

    step_obs = int(1 / dt)
    step = 0
    x = x0

    ts = [0.0]
    xs = [x]
    ys = []
    y_steps = []
    for step in range(0, iter_max):
        t = step * dt
        x += dx(x, dt, gamma=gamma, sigma=sigma)

        if step % step_obs == 0:
            ys.append(y(x, std_ob=std_ob))
            y_steps.append(step)

        ts.append(t)
        xs.append(x)
    
    return np.array(ts), np.array(xs), np.array(ys), np.array(y_steps)
