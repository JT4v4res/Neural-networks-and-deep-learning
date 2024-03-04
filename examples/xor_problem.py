import numpy as np
from multi_layer_neural_network.neural_network import NeuralNetwork, NetworkLayer
from loss_functions.loss_funcs import binary_cross_entropy_error
from activation_functions.activation_functions import sigmoid
from sklearn.metrics import accuracy_score
from utils import plot
import matplotlib.pyplot as plt


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0]).reshape(-1, 1)

input_dim, output_dim = x.shape[1], y.shape[1]

neural_network = NeuralNetwork(cost_func=binary_cross_entropy_error, learning_rate=0.5)
neural_network.layers.append(NetworkLayer(input_dim, 3, activation=sigmoid))
neural_network.layers.append(NetworkLayer(3, output_dim, activation=sigmoid))

neural_network.fit(x, y, epochs=5000, verbose=1000)

y_pred = neural_network.predict(x)

print('XOR problem accuracy: {:.2f}%'.format(100 * accuracy_score(y, y_pred > 0.5)))

plot.classification_predictions(x, y, is_binary=True, nn=neural_network, threshold=0.5, cmap='bwr')

plt.title('XOR problem')

plt.show()
