import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

class Model:
    def __init__(self, k, var_st, var_ob):
        self.k = k
        self.var_st = var_st
        self.var_ob = var_ob
