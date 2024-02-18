import numpy as np


def mean_squared_error(y, y_pred, derivative=False):
    if derivative:
        return -(y - y_pred) / y.shape[0]
    return 0.5 * np.mean((y - y_pred) ** 2)


predicted = np.array([0, 1, 1, 0])
actual = np.array([0, 1, 1, 1])

for i in range(0, 4):
    print(mean_squared_error(actual[i], predicted[i]))
