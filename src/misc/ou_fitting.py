import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model


def realize(x0, gamma, sigma, T, dt, N):

    x = np.repeat(x0, N)
    xs = [x.copy()]
    ts = [0]
    for step in range(T):
        t = dt * step
        a = -gamma * x
        b = sigma
        dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=N)

        dx = a * dt + b * dW
        x += dx

        xs.append(x.copy())
        ts.append(t)

    return np.array(ts), np.array(xs)


def make_mu_1step(x0, gamma, sigma, dt, N):
    _, xs = realize(x0, gamma, sigma, int(1/dt), dt, N)
    return xs.mean(axis=1)[-1], np.square(xs).mean(axis=1)[-1]


x = np.random.uniform(0, 1, 100)
moments = np.array([make_mu_1step(x, 1/20, 1.0, 0.01, 1000) for x in x])
y = moments[:, 0]

model = linear_model.LinearRegression() 
X = x[:,np.newaxis]
reg = model.fit(X, y)
print(reg.coef_)
print(np.exp(-1/20))
print(reg.intercept_)

# vars = moments[:, 1] - np.square(moments[:, 0])
# reg_t = model.fit(X, vars)
# print(reg_t.coef_)
# print(1/(2*(1/20)) * (1 - np.exp(-2/20)))
# print(reg_t.intercept_)



fig: plt.Figure = plt.figure()
ax: plt.Axes = fig.add_subplot()

plot_x = np.linspace(0, 1, 100)
plot_y = reg.predict(plot_x[:,np.newaxis])

ax.set_xlabel("x")
ax.set_ylabel("S(x)")
ax.set_title("mean data of OU process")
ax.scatter(x, y, label="data")
ax.plot(plot_x, plot_y, label="estimate")
ax.legend()
# ax.scatter(x, moments[:, 1] - np.square(moments[:, 0]))

plt.show()
