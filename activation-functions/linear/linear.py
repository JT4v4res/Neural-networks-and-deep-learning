import numpy as np


def linear(x, derivative=False):
    if derivative:
        return np.ones_like(x)
    return x


print(linear(-10))
print(linear(0))
print(linear(10))
