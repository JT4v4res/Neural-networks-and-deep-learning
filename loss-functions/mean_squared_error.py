import numpy as np


def mean_squared_error(output_vector):
    return (output_vector.sum()) / len(output_vector)


predicted = np.array([0, 1, 1, 0])
actual = np.array([0, 1, 1, 1])

error = actual - predicted

for i in error:
    i = i ** 2

print(mean_squared_error(error))
