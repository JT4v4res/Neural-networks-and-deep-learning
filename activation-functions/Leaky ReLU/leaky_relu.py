import numpy as np


def leaky_relu(x, derivative=False):
    alpha = 0.1
    if derivative:
        np.where(x <= 0, alpha, 1)
    return np.where(x <= 0, alpha * x, x)


test = np.array([-10, 0, 10])
for i in test:
    print(f'Value for x = {i} is Leaky ReLU: {leaky_relu(i)}')
