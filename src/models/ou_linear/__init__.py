import numpy as np
from ..linear import Model as LinearModel

class Model:
    def __init__(self, gamma, sigma, l, var_ob):
        self.gamma = gamma
        self.sigma = sigma
        self.l = l
        self.var_ob = var_ob

    def to_linear_model(self):
        k = np.exp(-self.gamma)
        var_st = self.sigma**2 * (1 - k**2) / (2*self.gamma)
        return LinearModel(k, var_st, self.l, self.var_ob)


def realize(x0, iter_max, *, dt = 0.01, model: Model):
    gamma = model.gamma
    sigma = model.sigma
    l = model.l
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

        dW = np.random.normal(0, dt)
        dx = -gamma * x * dt + sigma * dW
        x += dx

        if step % step_obs == 0:
            y = l*x + np.random.normal(0, std_ob)
            ys.append(y)
            y_steps.append(step)

        ts.append(t)
        xs.append(x)
    
    return np.array(ts), np.array(xs), np.array(ys), np.array(y_steps)
