import numpy as np
from multi_layer_neural_network.neural_network import NeuralNetwork, NetworkLayer
from loss_functions.loss_funcs import binary_cross_entropy_error
from activation_functions.activation_functions import sigmoid, relu
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles
from utils import plot
import matplotlib.pyplot as plt


x, y = make_circles(n_samples=500, noise=0.1, factor=0.4, random_state=1234)
y = y.reshape(-1, 1)

input_dim, output_dim = x.shape[1], y.shape[1]

neural_network = NeuralNetwork(cost_func=binary_cross_entropy_error, learning_rate=0.3)
neural_network.layers.append(NetworkLayer(input_dim, 8, activation=relu))
neural_network.layers.append(NetworkLayer(8, output_dim, activation=sigmoid))

neural_network.fit(x, y, epochs=5000, verbose=1000)

y_pred = neural_network.predict(x)

plot.classification_predictions(x, y, is_binary=True, nn=neural_network, threshold=0.5, cmap='bwr')

plt.title('Binary classification problem with circles')

print('Binary classification with circles accuracy: {:.2f}%'.format(100 * accuracy_score(y, y_pred > 0.5)))

plt.show()
