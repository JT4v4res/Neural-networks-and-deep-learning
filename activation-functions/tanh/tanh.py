import numpy as np


def tanh(x, derivative=False):
    if derivative:
        y = tanh(x)
        return 1 - y ** 2
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


input = [-10, 0, 10]

for i in input:
    print(f'x = {i} tanh = {tanh(i)}')
