from matplotlib import pyplot as plt
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .common import feed_csv


@dataclass
class Result:
    t: NDArray[np.float64]
    hgm: NDArray[np.float64] 
    particle: NDArray[np.float64]


with open('data/ou-iori_t_abserrs_N1000_rtol1e-31e-31e-5_particle40.csv') as f:
    result_err = feed_csv(f)


print(result_err.keys())


result = Result(result_err['t'],
                result_err['err_hgm'],
                result_err['err_particle'])


def plot(ax: plt.Axes):
    ax.plot(result.t, result.particle, label="Particle", c='forestgreen')
    ax.plot(result.t, result.hgm, label="Proposed", c='red')
    ax.set_xlabel('time')
    ax.set_ylabel('|(estimate) - (true state)|')
    ax.legend()


fig: plt.Figure = plt.figure(figsize=(6.4, 3.2))
fig.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.15)
ax = fig.add_subplot()
plot(ax)
plt.show()
