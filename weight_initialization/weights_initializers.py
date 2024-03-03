import numpy as np


def zeros(rows, cols):
    return np.zeros((rows, cols))


def ones(rows, cols):
    return np.ones((rows, cols))


def random_normal(rows, cols):
    return np.random.randn(rows, cols)


def random_uniform(rows, cols):
    return np.random.rand(rows, cols)


def normal_glorot(rows, cols):
    std_dev = np.sqrt(2. / (rows + cols))
    return std_dev * np.random.randn(rows, cols)


def uniform_glorot(rows, cols):
    std_dev = np.sqrt(6. / (rows + cols))
    return 2 * std_dev * np.random.rand(rows, cols) - std_dev
