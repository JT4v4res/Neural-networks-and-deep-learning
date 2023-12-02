import numpy as np


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
                print(neuron_inputs[i], ' * ', self.weights[j])
                self.neurons[j] = sigmoid(np.dot(neuron_inputs[i], self.weights[j]) + bias)

        return self.neurons


class NeuralNetwork:
    def __init__(self, input_len=0):
        self.neuron_layers = []
        self.total_layers = 0
        self.bias = 0

    def train(self, epoch, entry, correct):
        for i in range(0, epoch):
            feed_input = entry
            for z in range(0, len(self.neuron_layers)):
                feed_input = self.neuron_layers[z].affine_function(np.asarray(feed_input), self.bias)

            answer = feed_input.sum()
            print(answer)


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 1, 1, 0])

neuralNetwork = NeuralNetwork(2)
neuralNetwork.neuron_layers.append(NetworkLayer(3, 2))

neuralNetwork.train(1, inputs, outputs)
