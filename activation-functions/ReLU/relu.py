import numpy as np


def relu(x, derivative=False):
    if derivative:
        return np.where(x <= 0, 0, 1)

    return np.maximum(0, x)


test = np.array([-10, 0, 10])

for i in test:
    print(f'Relu on x = {i} gives: {relu(i)}')
