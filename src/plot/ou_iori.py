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
    particle: NDArray[np.float64] 
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
                result_estimate['particle'],
                result_estimate['ekf'])


def plot(ax: plt.Axes):
    ax.plot(result.t, result.x, label="state", c='lightgray')
    ax.plot(result.t_disc, result.hgm, label='HGM')
    ax.plot(result.t_disc, result.particle, label='Particle')
    ax.plot(result.t_disc, result.ekf, label='EKF')
    ax.legend()


fig = plt.figure()
ax = fig.add_subplot()
plot(ax)
plt.show()
