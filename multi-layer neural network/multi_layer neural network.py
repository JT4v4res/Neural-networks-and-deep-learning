import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import accuracy_score
from utils import plot
from sklearn.preprocessing import OneHotEncoder
from learning_rate_decay.lr_decay import none_decay
from activation_functions.activation_functions import sigmoid, linear, relu, tanh
from loss_functions.loss_funcs import softmax_neg_log_likelihood, binary_cross_entropy_error, mean_squared_error
from regularization.regularizations import l1_regularization, l2_regularization
from weight_initialization.weights_initializers import random_normal, random_uniform, zeros, ones
import _pickle as pkl


def batch_sequencial(x, y, batch_size=None):
    batch_size = x.shape[0] if batch_size is None else batch_size
    n_batches = x.shape[0] // batch_size

    for batch in range(n_batches):
        offset = batch_size * batch
        x_batch, y_batch = x[offset:offset + batch_size], y[offset:offset + batch_size]
        yield (x_batch, y_batch)


def shuffle_batch(x, y, batch_size=None):
    shuffle_idx = np.random.permutation(range(x.shape[0]))
    return batch_sequencial(x[shuffle_idx], y[shuffle_idx], batch_size)


def batch_normalization_forward(layer, x, training=True):
    mu = np.mean(x, axis=0) if training else layer._pop_mean
    var = np.var(x, axis=0) if training else layer._pop_var
    x_norm = (x - mu) / np.sqrt(var + 1e-8)
    out = layer.gamme * x_norm + layer.beta

    if training:
        layer._pop_mean = layer.batch_norm_decay * layer._pop_mean + (1.0 - layer.batch_norm_decay) * mu
        layer._pop_var = layer.batch_norm_decay * layer._pop_var + (1.0 - layer.batch_norm_decay) * var
        layer._bn_cache = (x, x_norm, mu, var)

    return out


def batch_normalization_backward(layer, dactivation):
    x, x_norm, mu, var = layer._bn_cache

    m = layer._activation_input.shape[0]
    x_mu = x - mu
    std_inv = 1.0 / np.sqrt(var + 1e-8)

    dx_norm = dactivation * layer.gamma
    dvar = np.sum(dx_norm * x_mu, axis=0) * -0.5 * (std_inv ** 3)
    dmu = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2.0 * x_mu, axis=0)

    dx = (dx_norm * std_inv) + (dvar * 2.0 * x_mu / m) + (dmu / m)
    layer._dgamma = np.sum(dactivation * x_norm, axis=0)
    layer._dbeta = np.sum(dactivation, axis=0)

    return dx


class NetworkLayer:
    def __init__(self, input_shape, output_shape, activation=linear, weight_initializer=random_normal,
                 bias_initializer=ones, dropout_prob=0.0, regularization=l2_regularization, reg_strength=0.0,
                 batch_norm=False, batch_norm_decay=0.9, is_trainable=True):
        self.input = None
        self.weights = weight_initializer(output_shape, input_shape)
        self.biases = bias_initializer(1, output_shape)
        self.activation = activation
        self.dropout_prob = dropout_prob
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.gamma, self.beta = ones(1, output_dim), zeros(1, output_dim)
        self.is_trainable = is_trainable

        self._activation_input, self._activation_output = None, None
        self._dweights, self._dbiases, self._prev_dweights = None, None, 0.0
        self._dropout_mask = None
        self._dgamma, self._dbeta = None, None
        self._pop_mean, self._pop_var = zeros(1, output_dim), zeros(1, output_dim)
        self._bn_cache = None


class NeuralNetwork:
    def __init__(self, cost_func=mean_squared_error, learning_rate=1e-3, lr_decay=none_decay.none_decay,
                 lr_decay_rate=0.0, lr_decay_steps=1, momentum=0.0, patience=np.inf):
        self.layers = []
        self.cost_func = cost_func
        self.learning_rate = self.start_lr = learning_rate
        self.lr_decay = lr_decay
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        self.momentum = momentum
        self.patience, self.waiting = patience, 0
        self._best_model, self._best_loss = self.layers, np.inf

    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=100,
            verbose=10, batch_gen=batch_sequencial, batch_size=None):
        x_val, y_val = (x_train, y_train) if (x_val is None) or (y_val is None) else (x_val, y_val)

        for epoch in range(0, epochs + 1):
            self.learning_rate = self.lr_decay(self.start_lr, epoch, self.lr_decay_rate, self.lr_decay_steps)

            for x_batch, y_batch in batch_gen(x_train, y_train, batch_size):
                y_pred = self.__feedforward(x_batch)
                self.__backpropagation(y_batch, y_pred)

            loss_val = self.cost_func(y_val, self.predict(x_val))
            if loss_val < self._best_loss:
                self._best_model, self._best_loss = self.layers, loss_val
                self.waiting = 0
            else:
                self.waiting += 1
                if self.waiting > self.patience:
                    self.layers = self._best_model
                    return

            if epoch % verbose == 0:
                loss_train = self.cost_func(y_train, self.predict(x_train))
                loss_reg = (1.0) / y_train.shape[0] * np.sum(
                    [layer.regularization(layer.weights) * layer.reg_strength for layer in self.layers])
                print(f'Epoch: {epoch}/{epochs}, Loss: {loss_train}, Regularization loss: {loss_reg}')

    def predict(self, x):
        return self.__feedforward(x, False)

    def save(self, file_path):
        pkl.dump(self, open(file_path, 'wb'), -1)

    def load(file_path):
        return pkl.load(open(file_path, 'rb'))

    def __feedforward(self, x, is_training=True):
        self.layers[0].input = x

        for layer, next_layer in zip(self.layers, self.layers[1:] + [NetworkLayer(0, 0)]):
            y = np.dot(layer.input, layer.weights.T) + layer.biases
            y = batch_normalization_forward(layer, y, is_training) if layer.batch_norm else y
            layer._dropout_mask = np.random.binomial(1, 1.0 - layer.dropout_prob, y.shape) / (1 - layer.dropout_prob)
            layer._activation_input = y
            layer._activation_output = next_layer.input = layer.activation(y) * (layer._dropout_mask if is_training else 1.0)

        return self.layers[-1]._activation_output

    def __backpropagation(self, y, y_pred):
        last_delta = self.cost_func(y, y_pred, True)

        for layer in reversed(self.layers):
            dactivation = layer.activation(layer._activation_input, True) * last_delta * layer._dropout_mask
            dactivation = batch_normalization_backward(layer, dactivation) if layer.batch_norm else dactivation
            last_delta = np.dot(dactivation, layer.weights)
            layer._dweights = np.dot(dactivation.T, layer.input)
            layer._dbiases = 1.0 * dactivation.sum(axis=0, keepdims=True)

        for layer in reversed(self.layers):
            if layer.is_trainable:
                layer._dweights = layer._dweights + (1.0 / y.shape[0]) * layer.reg_strength * layer.regularization(
                    layer.weights, True)
                layer._prev_dweights = - self.learning_rate * layer._dweights + self.momentum * layer._prev_dweights
                layer.weights = layer.weights - self.learning_rate * layer._dweights
                layer.biases = layer.biases - self.learning_rate * layer._dbiases
                if layer.batch_norm:
                    layer.gamma = layer.gamma - self.learning_rate * layer._dgamma
                    layer.beta = layer.beta - self.learning_rate * layer._dbeta


# The XOR problem
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0]).reshape(-1, 1)

input_dim, output_dim = x.shape[1], y.shape[1]

neural_network = NeuralNetwork(cost_func=binary_cross_entropy_error, learning_rate=0.5)
neural_network.layers.append(NetworkLayer(input_dim, 3, activation=sigmoid))
neural_network.layers.append(NetworkLayer(3, output_dim, activation=sigmoid))

neural_network.fit(x, y, epochs=5000, verbose=1000)

y_pred = neural_network.predict(x)

print(f'XOR accuracy: {accuracy_score(y, y_pred > 0.5) * 100}%')

plot.classification_predictions(x, y, is_binary=True, nn=neural_network, threshold=0.5, cmap='bwr')

plt.title('XOR problem')

plt.show()

# Binary Classification

x, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=1234)
y = y.reshape(-1, 1)

input_dim, output_dim = x.shape[1], y.shape[1]

neural_network = NeuralNetwork(cost_func=binary_cross_entropy_error, learning_rate=0.1)
neural_network.layers.append(NetworkLayer(input_dim, output_dim, activation=sigmoid))

neural_network.fit(x, y, 5000, verbose=1000)

print(f'Binary classification accuracy: {100 * accuracy_score(y, np.array(neural_network.predict(x) > 0.5))}%')

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

neural_network.fit(x, y, 5000, verbose=1000)

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

neural_network.fit(x, y, epochs=5000, verbose=1000)

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
neural_network.layers.append(NetworkLayer(input_dim, 8, activation=tanh))
neural_network.layers.append(NetworkLayer(8, 6, activation=tanh))
neural_network.layers.append(NetworkLayer(6, output_dim, activation=sigmoid))

neural_network.fit(x, y, epochs=5000, verbose=1000)

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

neural_network.fit(x, y_onehot, 5000, verbose=1000)

print(
    f'Multiclass classification 3 clusters accuracy: {100 * accuracy_score(y, np.argmax(neural_network.predict(x), axis=1))}%')

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
neural_network.layers.append(NetworkLayer(input_dim, 8, activation=relu))
neural_network.layers.append(NetworkLayer(8, 8, activation=relu))
neural_network.layers.append(NetworkLayer(8, output_dim, activation=linear))

neural_network.fit(x, y_onehot, 5000, verbose=1000)

print(
    f'Multiclass classification 4 clusters accuracy: {100 * accuracy_score(y, np.argmax(neural_network.predict(x), axis=1))}%')

plot.classification_predictions(x, y, is_binary=False, nn=neural_network, threshold=0.5, cmap='viridis')

plt.title('Multiclass 4 clusters')

plt.show()

# Moons
x, y = make_moons(n_samples=500, noise=0.2, random_state=1234)
y = y.reshape(-1, 1)

input_dim, output_dim = x.shape[1], y.shape[1]

neural_network = NeuralNetwork(cost_func=binary_cross_entropy_error, learning_rate=0.5)
neural_network.layers.append(
    NetworkLayer(input_dim, 8, activation=tanh, reg_strength=0.3, regularization=l2_regularization))
neural_network.layers.append(
    NetworkLayer(8, output_dim, activation=sigmoid, reg_strength=0.3, regularization=l2_regularization))

neural_network.fit(x, y, epochs=5000, verbose=1000)

y_pred = neural_network.predict(x)

plot.classification_predictions(x, y, is_binary=True, nn=neural_network, threshold=0.5, cmap='bwr')

plt.title('Binary classification problem with moons with regularization')

print('Acurácia with moons: {:.2f}%'.format(100 * accuracy_score(y, y_pred > 0.5)))

plt.show()

# Moons
x, y = make_moons(n_samples=500, noise=0.2, random_state=1234)
y = y.reshape(-1, 1)

input_dim, output_dim = x.shape[1], y.shape[1]

neural_network = NeuralNetwork(cost_func=binary_cross_entropy_error, learning_rate=0.5, momentum=0.3)
neural_network.layers.append(NetworkLayer(input_dim, 8, activation=tanh))
neural_network.layers.append(NetworkLayer(8, output_dim, activation=sigmoid))

neural_network.fit(x, y, epochs=5000, verbose=500)

y_pred = neural_network.predict(x)

plot.classification_predictions(x, y, is_binary=True, nn=neural_network, threshold=0.5, cmap='bwr')

plt.title('Binary classification problem with moons with momentum')

print('Acurácia with moons: {:.2f}%'.format(100 * accuracy_score(y, y_pred > 0.5)))

plt.show()
