import numpy as np
from models.ou_iori import phi
from models.ou_iori import Model, realize
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

phi0 = [0.000352906346645659, 0.0002628470890941869, 0.005425863723318185, 4.465619573290613e-05, 0.001271863762133951, -0.0006375520335499671, 0.29367085906672813]
phi1 = [-0.05225524503399767, -0.01802159748874855, 0.001136691630898873, 0.24570465210777287, 0.2818491908373402, -0.0007028978820531912, 0.005275708109337688]
phi2 = [0.001280434006323272, -0.0011272004604712114, 0.6275226423490808, 0.002068901759211261, 0.01044308984043396, -0.3032751969505906, 0.5501203329514268]


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
