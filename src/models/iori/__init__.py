import numpy as np


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

