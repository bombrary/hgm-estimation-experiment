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


with open('data/ou-iori_t_abserrs_N1000_rtol1e-31e-31e-5_particle100.csv') as f:
    result_err = feed_csv(f)


print(result_err.keys())


result = Result(result_err['ts'],
                result_err['hgm'],
                result_err['particle'])


def plot(ax: plt.Axes):
    ax.plot(result.t, result.hgm, label="hgm")
    ax.plot(result.t, result.particle, label="particle")
    ax.legend()


fig = plt.figure()
ax = fig.add_subplot()
plot(ax)
plt.show()
