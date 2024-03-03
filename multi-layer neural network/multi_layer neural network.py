import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import accuracy_score
from utils import plot
from sklearn.preprocessing import OneHotEncoder


def sigmoid(x, derivative=False):
    if derivative:
        y = sigmoid(x)
        return y * (1 - y)

    return 1 / (1 + np.exp(-x))


def linear(x, derivative=False):
    if derivative:
        return np.ones_like(x)
    return x


def tanh(x, derivative=False):
    if derivative:
        y = tanh(x)
        return 1 - y ** 2
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x, derivative=False):
    if derivative:
        return np.where(x <= 0, 0, 1)

    return np.maximum(0, x)


def leaky_relu(x, derivative=False):
    alpha = 0.1
    if derivative:
        np.where(x <= 0, alpha, 1)
    return np.where(x <= 0, alpha * x, x)


def elu(x, derivative=False):
    alpha = 1.0
    if derivative:
        y = elu(x)
        return np.where(x <= 0, y + alpha, 1)
    return np.where(x <= 0, alpha * (np.exp(x) - 1), x)


def softmax(x, y_oh=None, derivative=False):
    if derivative:
        y_pred = softmax(x)
        k = np.nonzero(y_pred * y_oh)
        pk = y_pred[k]
        y_pred[k] = pk * (1.0 - pk)
        return y_pred
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def neg_log_likelihood(y_oh, y_pred, derivative=False):
    k = np.nonzero(y_pred * y_oh)
    pk = y_pred[k]
    if derivative:
        y_pred[k] = (-1.0 / pk)
        return y_pred
    return np.mean(-np.log(pk))


def softmax_neg_log_likelihood(y_oh, y_pred, derivative=False):
    y_softmax = softmax(y_pred)
    if derivative:
        return -(y_oh - y_softmax) / y_oh.shape[0]
    return neg_log_likelihood(y_oh, y_softmax)


def mean_absolute_error(y, y_pred, derivative=False):
    if derivative:
        return np.where(y_pred > y, 1, -1) / y.shape[0]
    return np.mean(np.abs(y - y_pred))


def mean_squared_error(y, y_pred, derivative=False):
    if derivative:
        return -(y - y_pred) / y.shape[0]
    return 0.5 * np.mean((y - y_pred) ** 2)


def binary_cross_entropy_error(y, y_pred, derivative=False):
    if derivative:
        return -(y - y_pred) / (y_pred * (1 - y_pred) * y.shape[0])
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def sigmoid_cross_entropy(y, y_pred, derivative=False):
    y_sigmoid = sigmoid(y_pred)
    if derivative:
        return -(y - y_sigmoid) / (y_sigmoid * (1 - y_sigmoid) * y.shape[0])
    return -np.mean(y * np.log(y_sigmoid) + (1 - y) * np.log(1 - y_sigmoid))


class NetworkLayer:
    def __init__(self, input_shape, output_shape, activation=linear):
        self.input = None
        self.weights = np.random.randn(output_shape, input_shape)
        self.biases = np.random.randn(1, output_shape)
        self.activation = activation

        self._activation_input, self._activation_output = None, None
        self._dweights, self._dbiases = None, None


class NeuralNetwork:
    def __init__(self, cost_func=mean_squared_error, learning_rate=1e-3):
        self.layers = []
        self.cost_func = cost_func
        self.learning_rate = learning_rate

    def fit(self, x_train, y_train, epochs=100, verbose=10):
        for epoch in range(0, epochs + 1):
            y_pred = self.__feedforward(x_train)
            self.__backpropagation(y_train, y_pred)

            if epoch % verbose == 0:
                loss_train = self.cost_func(y_train, self.predict(x_train))
                print(f'Epoch: {epoch}/{epochs}, Loss: {loss_train}')

    def predict(self, x):
        return self.__feedforward(x)

    def __feedforward(self, x):
        self.layers[0].input = x

        for layer, next_layer in zip(self.layers, self.layers[1:] + [NetworkLayer(0, 0)]):
            y = np.dot(layer.input, layer.weights.T) + layer.biases
            layer._activation_input = y
            layer._activation_output = next_layer.input = layer.activation(y)

        return self.layers[-1]._activation_output

    def __backpropagation(self, y, y_pred):
        last_delta = self.cost_func(y, y_pred, True)

        for layer in reversed(self.layers):
            dactivation = layer.activation(layer._activation_input, True) * last_delta
            last_delta = np.dot(dactivation, layer.weights)
            layer._dweights = np.dot(dactivation.T, layer.input)
            layer._dbiases = 1.0 * dactivation.sum(axis=0, keepdims=True)

        for layer in reversed(self.layers):
            layer.weights = layer.weights - self.learning_rate * layer._dweights
            layer.biases = layer.biases - self.learning_rate * layer._dbiases


# The XOR problem
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0]).reshape(-1, 1)

input_dim, output_dim = x.shape[1], y.shape[1]

neural_network = NeuralNetwork(cost_func=binary_cross_entropy_error, learning_rate=0.5)
neural_network.layers.append(NetworkLayer(input_dim, 2, activation=sigmoid))
neural_network.layers.append(NetworkLayer(2, output_dim, activation=sigmoid))

neural_network.fit(x, y, 1000, verbose=100)

y_pred = neural_network.predict(x)

print(f'XOR accuracy: {accuracy_score(y, y_pred > 0.5)}')

plot.classification_predictions(x, y, is_binary=True, nn=neural_network, threshold=0.5, cmap='bwr')

plt.title('XOR problem')

plt.show()

# Binary Classification

x, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=1234)
y = y.reshape(-1, 1)

input_dim, output_dim = x.shape[1], y.shape[1]

neural_network = NeuralNetwork(cost_func=binary_cross_entropy_error, learning_rate=0.1)
neural_network.layers.append(NetworkLayer(input_dim, output_dim, activation=sigmoid))

neural_network.fit(x, y, 1000, verbose=100)

print(f'Binary classification accuracy: {100 * accuracy_score(y, np.array(neural_network.predict(x) > 0.5))}')

plot.classification_predictions(x, y, is_binary=True, nn=neural_network, threshold=0.5, cmap='bwr')

plt.title('Binary classification problem')

plt.show()

# Binary class. with 4 clusters

x, y = make_blobs(n_samples=500, n_features=2, centers=[(-3, -3), (3, 3), (-3, 3), (3, -3)], random_state=1234)
y = y.reshape(-1, 1)
y = np.where(y >= 2, 1, 0)

input_dim, output_dim = x.shape[1], y.shape[1]

neural_network = NeuralNetwork(cost_func=binary_cross_entropy_error, learning_rate=0.1)
neural_network.layers.append(NetworkLayer(input_dim, 4, activation=sigmoid))
neural_network.layers.append(NetworkLayer(4, output_dim, activation=sigmoid))

neural_network.fit(x, y, 1000, verbose=100)

y_pred = neural_network.predict(x)
print('Acurácia with 4 clusters: {:.2f}%'.format(100 * accuracy_score(y, y_pred > 0.5)))

plot.classification_predictions(x, y, is_binary=True, nn=neural_network, threshold=0.5, cmap='bwr')

plt.title('Binary classification problem with 4 clusters')

plt.show()

# Circles

x, y = make_circles(n_samples=500, noise=0.1, factor=0.4, random_state=1234)
y = y.reshape(-1, 1)

input_dim, output_dim = x.shape[1], y.shape[1]

neural_network = NeuralNetwork(cost_func=binary_cross_entropy_error, learning_rate=0.5)
neural_network.layers.append(NetworkLayer(input_dim, 4, activation=relu))
neural_network.layers.append(NetworkLayer(4, output_dim, activation=sigmoid))

neural_network.fit(x, y, epochs=1000, verbose=100)

y_pred = neural_network.predict(x)

plot.classification_predictions(x, y, is_binary=True, nn=neural_network, threshold=0.5, cmap='bwr')

plt.title('Binary classification problem with circles')

print('Acurácia with circles: {:.2f}%'.format(100 * accuracy_score(y, y_pred > 0.5)))

plt.show()

# Moons
x, y = make_moons(n_samples=500, noise=0.2, random_state=1234)
y = y.reshape(-1, 1)

input_dim, output_dim = x.shape[1], y.shape[1]

neural_network = NeuralNetwork(cost_func=binary_cross_entropy_error, learning_rate=0.5)
neural_network.layers.append(NetworkLayer(input_dim, 4, activation=relu))
neural_network.layers.append(NetworkLayer(4, 4, activation=relu))
neural_network.layers.append(NetworkLayer(4, output_dim, activation=sigmoid))

neural_network.fit(x, y, epochs=1000, verbose=100)

y_pred = neural_network.predict(x)

plot.classification_predictions(x, y, is_binary=True, nn=neural_network, threshold=0.5, cmap='bwr')

plt.title('Binary classification problem with moons')

print('Acurácia with moons: {:.2f}%'.format(100 * accuracy_score(y, y_pred > 0.5)))

plt.show()

# Multiclass Classification 3 clusters

x, y = make_blobs(n_samples=300, n_features=3, centers=[(0, -3), (-3, 3), (3, 3)], random_state=1234)
y = y.reshape(-1, 1)

onehot = OneHotEncoder(sparse_output=False)
y_onehot = onehot.fit_transform(y)

input_dim, output_dim = x.shape[1], y_onehot.shape[1]

neural_network = NeuralNetwork(cost_func=softmax_neg_log_likelihood, learning_rate=0.1)
neural_network.layers.append(NetworkLayer(input_dim, 2, activation=relu))
neural_network.layers.append(NetworkLayer(2, output_dim, activation=linear))

neural_network.fit(x, y_onehot, 1000, verbose=100)

print(f'Multiclass classification 3 clusters accuracy: {100 * accuracy_score(y, np.argmax(neural_network.predict(x), axis=1))}')

plot.classification_predictions(x, y, is_binary=False, nn=neural_network, threshold=0.5, cmap='viridis')

plt.title('Multiclass 3 clusters')

plt.show()

# Multiclass Classification 4 clusters

x, y = make_blobs(n_samples=300, n_features=4, centers=[(-3, 0), (3, 0), (0, 3), (0, -3)], random_state=1234)
y = y.reshape(-1, 1)

onehot = OneHotEncoder(sparse_output=False)
y_onehot = onehot.fit_transform(y)

input_dim, output_dim = x.shape[1], y_onehot.shape[1]

neural_network = NeuralNetwork(cost_func=softmax_neg_log_likelihood, learning_rate=0.1)
neural_network.layers.append(NetworkLayer(input_dim, 2, activation=relu))
neural_network.layers.append(NetworkLayer(2, output_dim, activation=linear))

neural_network.fit(x, y_onehot, 1000, verbose=100)

print(f'Multiclass classification 4 clusters accuracy: {100 * accuracy_score(y, np.argmax(neural_network.predict(x), axis=1))}')

plot.classification_predictions(x, y, is_binary=False, nn=neural_network, threshold=0.5, cmap='viridis')

plt.title('Multiclass 4 clusters')

plt.show()

