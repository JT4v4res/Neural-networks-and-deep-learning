import numpy as np
from multi_layer_neural_network.neural_network import NeuralNetwork, NetworkLayer
from loss_functions.loss_funcs import softmax_neg_log_likelihood
from activation_functions.activation_functions import tanh, linear
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
from sklearn.preprocessing import OneHotEncoder
from utils import plot
import matplotlib.pyplot as plt


x, y = make_blobs(n_samples=300, n_features=3, centers=[(0, -3), (-3, 3), (3, 3)], random_state=42)
y = y.reshape(-1, 1)

onehot = OneHotEncoder(sparse_output=False)
y_onehot = onehot.fit_transform(y)

input_dim, output_dim = x.shape[1], y_onehot.shape[1]

neural_network = NeuralNetwork(cost_func=softmax_neg_log_likelihood, learning_rate=0.1)
neural_network.layers.append(NetworkLayer(input_dim, 4, activation=tanh))
neural_network.layers.append(NetworkLayer(4, output_dim, activation=linear))

neural_network.fit(x, y_onehot, epochs=5000, verbose=1000)

print('Multiclass classification 3 classes accuracy: {:.2f}%'.format(100 * accuracy_score(y, np.argmax(neural_network.predict(x), axis=1))))

plot.classification_predictions(x, y, is_binary=False, nn=neural_network, threshold=0.5, cmap='viridis')

plt.title('Multiclass 3 clusters')

plt.show()
