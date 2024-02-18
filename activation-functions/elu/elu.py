import numpy as np


def elu(x, derivative=False):
    alpha = 1.0
    if derivative:
        y = elu(x)
        return np.where(x <= 0, y + alpha, 1)
    return np.where(x <= 0, alpha * (np.exp(x) - 1), x)


print(elu(np.array([-10])))
print(elu(np.array([0])))
print(elu(np.array([10])))
