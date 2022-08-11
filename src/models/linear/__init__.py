from dataclasses import dataclass
import numpy as np
from fractions import Fraction


@dataclass
class Model:
    k: float
    var_st: float
    l: float
    var_ob: float

    def to_frac(self):
        return ( Fraction(self.k)
               , Fraction(self.var_st)
               , Fraction(self.l)
               , Fraction(self.var_ob)
               )


def realize(x0, n, *, model: Model):
    std_ob = np.sqrt(model.var_ob)
    std_st = np.sqrt(model.var_st)
    k = model.k
    l = model.l
    
    x = x0
    xs = [x0]
    ys = [l*x0 + np.random.normal(0, std_ob)]
    for _ in range(0, n-1):
        x = k*x + np.random.normal(0, std_st)
        y = l*x + np.random.normal(0, std_ob)
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)
