import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        y = sigmoid(x)
        return y * (1 - y)

    return 1 / (1 + np.exp(-x))


input = [-10, 0, 10]

for i in input:
    print(f'x = {i}, sigmoid value = {sigmoid(i)}')
