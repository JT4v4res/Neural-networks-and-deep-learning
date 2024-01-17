import numpy as np


def sigmoid_derivative(x):
    return (1 - sigmoid(x)) * sigmoid(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NetworkLayer:
    def __init__(self, layer_length=0, input_size=0):
        self.neurons = np.zeros(layer_length)
        self.weights = self.weight_initialize(input_size)

    def weight_initialize(self, input_size):
        weights = []

        for i in range(0, len(self.neurons)):
            weights.append(np.random.rand(input_size))

        return np.array(weights)

    def affine_function(self, neuron_inputs,  bias):

        for i in range(0, len(neuron_inputs)):
            for j in range(0, len(self.weights)):
                self.neurons[j] = sigmoid(np.sum(np.dot(neuron_inputs[i], self.weights[j])) + bias)

        return self.neurons


class NeuralNetwork:
    def __init__(self, input_len=0):
        self.neuron_layers = []
        self.total_layers = 0
        self.bias = 0
        self.loss_function = 0
        self.y_pred = []
        self.learning_rate = 0.1

    def train(self, epoch, entry):
        for i in range(0, epoch):
            feed_input = entry
            self.y_pred = []
            for z in range(0, len(self.neuron_layers)):
                feed_input = self.neuron_layers[z].affine_function(np.asarray(feed_input), self.bias)

            print(feed_input)

            delta_out = sigmoid_derivative(feed_input)

            last_deltas = dict()

            self.neuron_layers[-1].weights += (delta_out * self.learning_rate)

            last_deltas[len(self.neuron_layers) - 1] = delta_out

            for z in range(len(self.neuron_layers) - 2, -1, -1):
                for idw, w in enumerate(self.neuron_layers[z].weights):
                    idx = 0
                    last_deltas[z] = w * last_deltas[z + 1]
                    self.neuron_layers[z].weights[idw] = w + self.learning_rate * last_deltas[z + 1][idx]
                    idx += 1


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 1, 1, 0])

neuralNetwork = NeuralNetwork(2)
neuralNetwork.neuron_layers.append(NetworkLayer(3, 2))
neuralNetwork.neuron_layers.append(NetworkLayer(1, 3))

neuralNetwork.train(100, inputs)
