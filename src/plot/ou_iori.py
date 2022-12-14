from matplotlib import pyplot as plt
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .common import feed_csv


@dataclass
class Result:
    t: NDArray[np.float64]
    x: NDArray[np.float64] 
    t_disc: NDArray[np.float64] 
    hgm: NDArray[np.float64] 
    particle40: NDArray[np.float64] 
    particle400: NDArray[np.float64] 
    ekf: NDArray[np.float64] 


with open('data/ou-iori_t_x.csv') as f:
    result_state = feed_csv(f)


with open('data/ou-iori_t_y_estimates.csv') as f:
    result_estimate = feed_csv(f)


print(result_state.keys())
print(result_estimate.keys())


result = Result(result_state['t'],
                result_state['x'],
                result_estimate['t'],
                result_estimate['hgm'],
                result_estimate['particle40'],
                result_estimate['particle400'],
                result_estimate['ekf'])


def plot(ax: plt.Axes):
    ax.plot(result.t, result.x, label="state", c='lightgray')
    ax.plot(result.t_disc, result.ekf, label='EKF', c='steelblue')
    ax.plot(result.t_disc, result.particle40, label='Particle 40', c='forestgreen')
    ax.plot(result.t_disc, result.particle400, label='Particle 400', c='lime')
    ax.plot(result.t_disc, result.hgm, label='Proposed', c='red')
    ax.set_xlabel('time')
    ax.set_ylabel('value')
    ax.legend()


fig: plt.Figure = plt.figure(figsize=(6.4, 3.2))
fig.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.15)
ax = fig.add_subplot()
plot(ax)
plt.show()
