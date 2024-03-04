import numpy as np
from multi_layer_neural_network.neural_network import NeuralNetwork, NetworkLayer
from loss_functions.loss_funcs import binary_cross_entropy_error
from activation_functions.activation_functions import tanh, sigmoid
from sklearn.metrics import accuracy_score
from utils import plot
import matplotlib.pyplot as plt
from utils import samples_generator


x, y = samples_generator.make_spiral(n_samples=100, n_class=2, radius=5, laps=1.75)
y = y.reshape(-1, 1)

input_dim, output_dim = x.shape[1], y.shape[1]

neural_network = NeuralNetwork(cost_func=binary_cross_entropy_error, learning_rate=0.5)
neural_network.layers.append(NetworkLayer(input_dim, 10, activation=tanh))
neural_network.layers.append(NetworkLayer(10, 10, activation=tanh))
neural_network.layers.append(NetworkLayer(10, output_dim, activation=sigmoid))

neural_network.fit(x, y, epochs=5000, verbose=1000)

print('Binary classification with spiral accuracy: {:.2f}%'.format(100 * accuracy_score(y, neural_network.predict(x) > 0.5)))
plot.classification_predictions(x, y, is_binary=True, nn=neural_network, threshold=0.5, cmap='bwr')

plt.title('Binary classification with spirals')

plt.show()
