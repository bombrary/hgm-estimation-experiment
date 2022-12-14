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



def plot_mean(ax: plt.Axes, model, label: str):
    coef = model.coef_
    intercept = model.intercept_
    ord = len(coef)
    plot_x = np.linspace(0, 1, 100)
    plot_y = [np.dot(make_ord_array(x, ord), coef) + intercept for x in plot_x ]
    ax.plot(plot_x, plot_y, label=label)


def plot_var(ax: plt.Axes, model, label: str):
    coef = model.coef_
    intercept = model.intercept_
    ord = len(coef)
    plot_x = np.linspace(0, 1, 100)
    plot_y = [np.dot(make_ord_array(x, ord), coef) + intercept for x in plot_x ]
    ax.plot(plot_x, plot_y, label=label)


N = 1000
M = 1000
x = np.random.uniform(0, 1, N)
moments = np.array([make_mu_1step(x, 1, 1/2, 0.01, M) for x in x], dtype=np.float64)
means = moments[:, 0]
vars = moments[:, 1] - np.square(moments[:, 0])


mean_model_1d = fitting_nth_order(x, means, 1, False, 0)
mean_model_2d = fitting_nth_order(x, means, 2, False, 0)

var_model_2d = fitting_nth_order(x, vars, 2, False, 0)
var_model_3d = fitting_nth_order(x, vars, 3, False, 0)
# var_model_3d_lasso = fitting_nth_order(x, vars, 3, False, 1e-4)
var_model_4d = fitting_nth_order(x, vars, 4, False, 0)
# var_model_4d_lasso = fitting_nth_order(x, vars, 4, False, 1e-4)

print(f'     mean(1d): {mean_model_1d.coef_}')
print(f'     mean(2d): {mean_model_2d.coef_}')
print(f'      var(2d): {var_model_2d.coef_}')
print(f'      var(3d): {var_model_3d.coef_}')
# print(f'var(3d Lasso): {var_model_3d_lasso.coef_}')
print(f'      var(4d): {var_model_4d.coef_}')
# print(f'var(4d Lasso): {var_model_4d_lasso.coef_}')

figsize = (4, 3)
dpi = 100
fig: plt.Figure = plt.figure(figsize=figsize, dpi=dpi)
fig.subplots_adjust(left=0.155, bottom=0.15, right=0.99, top=0.975, wspace=0.24, hspace=0.23)
ax_mean = fig.add_subplot()

def setup_mean(ax: plt.Axes):
    ax.set_xlabel("x")
    ax.set_ylabel("S(x)")
    #ax.set_title("mean data of sFKPP process")

    ax.scatter(x, moments[:, 0], c="lightgray", label="data")

    plot_mean(ax, mean_model_1d, "1st order")
    plot_mean(ax, mean_model_2d, "2nd order")

    ax.legend()

setup_mean(ax_mean)

fig: plt.Figure = plt.figure(figsize=figsize, dpi=dpi)
fig.subplots_adjust(left=0.155, bottom=0.15, right=0.99, top=0.975, wspace=0.24, hspace=0.23)
ax_var = fig.add_subplot()

def setup_var(ax: plt.Axes):
    ax.set_xlabel("x")
    ax.set_ylabel("T(x)")
    ax.scatter(x, moments[:, 1] - np.square(moments[:, 0]), c="lightgray", label="data")
    #ax.set_title("var data of sFKPP process")

    plot_var(ax, var_model_2d, "2nd order")
    plot_var(ax, var_model_3d, "3rd order")
    # plot_var(ax, var_model_3d_lasso, "3rd order (LASSO)")
    plot_var(ax, var_model_4d, "4th order")
    # plot_var(ax, var_model_4d_lasso, "4th order (LASSO)")

    ax.legend()

setup_var(ax_var)

plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.show()
