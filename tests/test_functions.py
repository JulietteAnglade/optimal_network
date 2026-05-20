import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_solver import CobbDouglasUtility, CobbDouglasProduction

def test_utility():
    util = CobbDouglasUtility(
        alpha_c=np.array([[0.5, 0.3], [0.3, 0.5]]),
        alpha_l=np.array([0.2, 0.2]),
        alpha_f=np.array([0.1, 0.1])
    )
    c = np.array([2.0, 2.0])
    l = 1.0
    f = 1.0
    u = util(c, l, f, s=0)
    assert u > 0
    print("Utility test passed")

def test_production():
    prod = CobbDouglasProduction(
        A=np.array([[1.0, 1.0], [1.0, 1.0]]),
        aH=np.array([0.4, 0.4]),
        aL=np.array([0.2, 0.2]),
        aQ=np.array([[0.0, 0.2], [0.2, 0.0]])
    )
    y = prod(H=1.0, L=1.0, Q_inp=np.array([0.0, 1.0]), s=0, j=0)
    assert y > 0
    print("Production test passed")

if __name__ == "__main__":
    test_utility()
    test_production()
    print("All tests passed!")