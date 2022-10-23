import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from numpy.typing import NDArray


def realize(x0, gamma, sigma, T, dt, N):

    x = np.repeat(x0, N)
    xs = [x.copy()]
    ts = [0]
    for step in range(T):
        t = dt * step
        a = -gamma * x * (1 - x)
        b = sigma * np.sqrt(x * (1-x))
        dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=N)

        dx = a * dt + b * dW
        x += dx
        x = np.clip(x, 0, 1)

        xs.append(x)
        ts.append(t)

    return np.array(ts), np.array(xs)


def make_mu_1step(x0, gamma, sigma, dt, N):
    _, xs = realize(x0, gamma, sigma, int(1/dt), dt, N)
    return xs.mean(axis=1)[-1], np.square(xs).mean(axis=1)[-1]


N = 1000
M = 1000
x = np.random.uniform(0, 1, N)
moments = np.array([make_mu_1step(x, 1, 1/2, 0.01, M) for x in x], dtype=np.float64)
means = moments[:, 0]
vars = moments[:, 1] - np.square(moments[:, 0])


def make_ord_array(x, ord):
    return np.array([x**n for n in range(1, ord+1)])

def fitting_nth_order(xs, y, ord, intercept, alpha):
    if alpha == 0:
        model = linear_model.LinearRegression(fit_intercept=intercept)
    else:
        model = linear_model.Lasso(alpha=alpha, fit_intercept=intercept)

    X = np.array([ make_ord_array(x, ord) for x in xs])
    model.fit(X, y)
    return model


mean_model = fitting_nth_order(x, means, 2, False, 0)
var_model = fitting_nth_order(x, vars, 4, True, 1e-5)

fig: plt.Figure = plt.figure()
ax0 = fig.add_subplot(2, 1, 1)
ax1 = fig.add_subplot(2, 1, 2)


def plot_mean(ax: plt.Axes):
    ax.set_xlabel("x")
    ax.set_ylabel("S(x)")
    ax.set_title("mean data of sFKPP process")

    ax.scatter(x, moments[:, 0], label="data")

    coef = mean_model.coef_
    intercept = mean_model.intercept_
    ord = len(coef)
    plot_x = np.linspace(0, 1, 100)
    plot_y = [np.dot(make_ord_array(x, ord), coef) + intercept for x in plot_x ]
    ax.plot(plot_x, plot_y, color='red', label='fitting result')

    ax.legend()


def plot_var(ax: plt.Axes):
    ax.set_xlabel("x")
    ax.set_ylabel("T(x)")
    ax.set_title("var data of sFKPP process")

    ax.scatter(x, moments[:, 1] - np.square(moments[:, 0]), label="data")

    coef = var_model.coef_
    intercept = var_model.intercept_
    ord = len(coef)
    plot_x = np.linspace(0, 1, 100)
    plot_y = [np.dot(make_ord_array(x, ord), coef) + intercept for x in plot_x ]
    ax.plot(plot_x, plot_y, color='red', label='fitting result')

    ax.legend()


plot_mean(ax0)
plot_var(ax1)

print(f'mean: {mean_model.coef_}')
print(f'var: {var_model.coef_}')

plt.show()
