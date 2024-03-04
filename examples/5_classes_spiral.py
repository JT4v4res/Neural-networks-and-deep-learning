import numpy as np
from multi_layer_neural_network.neural_network import NeuralNetwork, NetworkLayer
from loss_functions.loss_funcs import softmax_neg_log_likelihood
from activation_functions.activation_functions import relu, linear
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
from sklearn.preprocessing import OneHotEncoder
from utils import plot
from utils import samples_generator
import matplotlib.pyplot as plt


x, y = samples_generator.make_spiral(n_samples=100, n_class=5, radius=1, laps=0.5)
y = y.reshape(-1, 1)

onehot = OneHotEncoder(sparse_output=False)
y_onehot = onehot.fit_transform(y)

input_dim, output_dim = x.shape[1], y_onehot.shape[1]

neural_network = NeuralNetwork(cost_func=softmax_neg_log_likelihood, learning_rate=0.3)
neural_network.layers.append(NetworkLayer(input_dim, 8, activation=relu))
neural_network.layers.append(NetworkLayer(8, 8, activation=relu))
neural_network.layers.append(NetworkLayer(8, output_dim, activation=linear))

neural_network.fit(x, y_onehot, epochs=5000, verbose=1000)

print('Multiclass classification 5 classes spiral accuracy: {:.2f}%'.format(100 * accuracy_score(y, np.argmax(neural_network.predict(x), axis=1))))


plot.classification_predictions(x, y, is_binary=False, nn=neural_network, threshold=0.5, cmap='viridis')

plt.title('Multiclass 4 clusters')

plt.show()
