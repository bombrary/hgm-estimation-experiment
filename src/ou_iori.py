import numpy as np
from hgm_estimation.models.ou_iori import phi
from hgm_estimation.models.ou_iori import Model, realize
import hgm_estimation.estimation as estimation
from setup import client
from fractions import Fraction
from utils import pfs_phi

gamma = 1/20
sigma = 1.0
var_ob = 1.0
model = Model(gamma, sigma, var_ob)

y0 = 0.01
mu0 = 0.01
lam0 = 1.0
# phi0s, phi1s, phi2s = phi.v_phis(y0, mu0, lam0, model=model)

phi0 = [0.0003469446951953614,
        0.0002636779683484747,
        0.005495603971894525,
        4.464462133313418e-05,
        0.0012718619490925676,
        -0.0006376324190870974,
        0.29367085906672813]
phi1 = [-0.05225485842641486,
        -0.018015753819322633,
        0.0011674688993323912,
        0.2457025698771216,
        0.28184932429967113,
        -0.0007029081067200238,
        0.005275708109337688]
phi2 = [0.001304512053934559,
        -0.0011102230246251565,
        0.627609075820601,
        0.002068859750181673,
        0.010443098885648538,
        -0.30328227956788467,
        0.5501203329514268]


client.load("asir-src/phi-ou-iori.rr")
client.execute_string(f'Pfs = gen_pfaffian({Fraction(model.k)}, {Fraction(model.var_st)}, {Fraction(var_ob)});')


x0 = 10.0
ts, xs, ys, y_steps = realize(x0, 5000, model=model)

result = estimation.run( y0, mu0, lam0, ys
              , v_phi00 = np.array(phi0)
              , v_phi10 = np.array(phi1)
              , v_phi20 = np.array(phi2)
              , pfs_phi0=lambda zs: pfs_phi(client, 0, zs)
              , pfs_phi1=lambda zs: pfs_phi(client, 1, zs)
              , pfs_phi2=lambda zs: pfs_phi(client, 2, zs))

client.send_shutdown()
