import numpy as np


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
    alpha=0.1
    if derivative:
        np.where(x <= 0, alpha, 1)
    return np.where(x <= 0, alpha * x, x)


def elu(x, derivative=False):
    alpha = 1.0
    if derivative:
        y = elu(x)
        return np.where(x <= 0, y + alpha, 1)
    return np.where(x <= 0, alpha * (np.exp(x) - 1), x)


def softmax(x, y_oh=None,derivative=False):
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


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 1, 1, 0])

input_dim, output_dim = inputs.shape[1], outputs.shape[0]

neural_network = NeuralNetwork(cost_func=mean_squared_error, learning_rate=0.7)
neural_network.layers.append(NetworkLayer(input_dim, 2, activation=sigmoid))
neural_network.layers.append(NetworkLayer(2, output_dim, activation=sigmoid))

neural_network.fit(inputs, outputs, 1000)

print(neural_network.predict(inputs))

