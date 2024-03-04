import numpy as np
from multi_layer_neural_network.neural_network import NeuralNetwork, NetworkLayer
from loss_functions.loss_funcs import mean_squared_error
from activation_functions.activation_functions import linear, tanh
import matplotlib.pyplot as plt
from utils import samples_generator


x, y = samples_generator.make_square(n_samples=100, x_min=-10, x_max=10, a=1, b=1, c=1, noise=10)

input_dim, output_dim = x.shape[1], y.shape[1]

neural_network = NeuralNetwork(cost_func=mean_squared_error, learning_rate=0.001)
neural_network.layers.append(NetworkLayer(input_dim, 10, activation=tanh))
neural_network.layers.append(NetworkLayer(10, 10, activation=tanh))
neural_network.layers.append(NetworkLayer(10, output_dim, activation=linear))

neural_network.fit(x, y, epochs=5000, verbose=1000)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x, y)
plt.plot(x, neural_network.predict(x), c='green')
plt.show()
