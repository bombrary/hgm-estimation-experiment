from ox_asir.client import Client
from matplotlib import pyplot as plt
import numpy as np


def plot_estimate(ax: plt.Axes, ts, mus, lams):
    sigs = np.sqrt(1 / lams)
    ax.plot(ts, mus, label="estimate")
    ax.fill_between(ts, mus + sigs, mus - sigs, color="lightgray")


def pfs_phi(client: Client, i, zs):
    y, mup, lamp = zs
    client.execute_string(f'subst(Pfs[{i}], y, {y}, mup, {mup}, lamp, {lamp});')
    return np.array(client.pop_cmo())
