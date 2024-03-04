import numpy as np


def elu(x, derivative=False):
    alpha = 1.0
    if derivative:
        y = elu(x)
        return np.where(x <= 0, y + alpha, 1)
    return np.where(x <= 0, alpha * (np.exp(x) - 1), x)


def leaky_relu(x, derivative=False):
    alpha = 0.1
    if derivative:
        np.where(x <= 0, alpha, 1)
    return np.where(x <= 0, alpha * x, x)


def linear(x, derivative=False):
    if derivative:
        return np.ones_like(x)
    return x


def relu(x, derivative=False):
    if derivative:
        return np.where(x <= 0, 0, 1)

    return np.maximum(0, x)


def sigmoid(x, derivative=False):
    if derivative:
        y = sigmoid(x)
        return y * (1 - y)

    return 1 / (1 + np.exp(-x))


def softmax(x, y_oh=None, derivative=False):
    if derivative:
        y_pred = softmax(x)
        k = np.nonzero(y_pred * y_oh)
        pk = y_pred[k]
        y_pred[k] = pk * (1.0 - pk)
        return y_pred
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def tanh(x, derivative=False):
    if derivative:
        y = tanh(x)
        return 1 - y ** 2
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
