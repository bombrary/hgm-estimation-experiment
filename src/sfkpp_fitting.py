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


M = 100
N = 1000
x = np.random.uniform(0, 1, M)
moments = np.array([make_mu_1step(x, 1, 1/5, 0.01, 1000) for x in x], dtype=np.float64)
y = moments[:, 0]

model = linear_model.LinearRegression() 
X = x[:,np.newaxis]
reg = model.fit(X, y)
print(f'coef: {reg.coef_}')
print(f'intercept: {reg.intercept_}')

# vars = moments[:, 1] - np.square(moments[:, 0])
# reg_t = model.fit(X, vars)
# print(reg_t.coef_)
# print(1/(2*(1/20)) * (1 - np.exp(-2/20)))
# print(reg_t.intercept_)



fig: plt.Figure = plt.figure()
ax0 = fig.add_subplot(2, 1, 1)
ax1 = fig.add_subplot(2, 1, 2)

# plot_x, plot_y = realize(0.8, 0.5, 0.5, 1000, 0.001, 3)
# ax.plot(plot_x, plot_y, label="plot")

plot_x = np.linspace(0, 1, 100)
plot_y = reg.predict(plot_x[:,np.newaxis])

def plot_mean(ax: plt.Axes):
    ax.set_xlabel("x")
    ax.set_ylabel("S(x)")
    ax.set_title("mean data of sFKPP process")
    ax.scatter(x, y, label="data")
    ax.plot(plot_x, plot_y, label="estimate")
    ax.legend()

def plot_var(ax: plt.Axes):
    ax.set_xlabel("x")
    ax.set_ylabel("S(x)")
    ax.set_title("var data of sFKPP process")
    ax.scatter(x, moments[:, 1] - np.square(moments[:, 0]), label="data")
    ax.legend()

plot_mean(ax0)
plot_var(ax1)

plt.show()
