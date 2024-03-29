# -*- coding: utf-8 -*-
"""Single Layer Perceptron implementation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19dHC1fbxaW25JtG2N8cpdk8ZcuBA3Xdl
"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.datasets import make_blobs
import numpy as np

X, y = make_blobs(centers=2, cluster_std=1, random_state=1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


class Perceptron:
    def __init__(self, learning_rate=0.01, entry_lenght=1):
        self.bias = 0
        self.weights = np.zeros(entry_lenght)
        self.learning_rate = learning_rate
        self.loss_function = 0

    def step_function(self, s):
        if s > 0:
            return 1
        return 0

    def affine_function(self, entry):
        s = entry.dot(self.weights) + self.bias

        return self.step_function(s)

    def weights_adjustment(self, entry):
        i = 0
        while i < len(entry):
            j = 0
            while j < len(self.weights):
                self.weights[j] = self.weights[j] + (self.learning_rate * self.loss_function * entry[j])
                j += 1
            i += 1

    def train(self, input, epoch, correct):
        current = 0
        while current <= epoch:
            pred = []
            i = 0
            while i < len(correct):
                answer = self.affine_function(np.asarray(input[i]))
                self.loss_function += correct[i] - answer
                self.weights_adjustment(input[i])
                self.bias += self.loss_function * self.learning_rate

                pred.append(answer == correct[i])

                count = pred.count(True)
                i += 1

            print(
                f'epoch {current}/{epoch}, loss: {(self.loss_function / 100):.2}, accuracy: {(count / len(correct)) * 100}%')
            current += 1


input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
correct = np.array([0, 1, 1, 1])

perceptron = Perceptron(entry_lenght=len(input[0]))

perceptron.train(input=input, epoch=10, correct=correct)

perceptron2 = Perceptron(entry_lenght=len(X[0]))

perceptron2.train(input=X, epoch=300, correct=y)

weights = perceptron2.weights
bias = perceptron2.bias

x_min, x_max = X.min(), X.max()
x1 = np.linspace(x_min, x_max)
y1 = -(weights[0] / weights[1]) * x1 - bias / weights[1]

plt.scatter(X[:,0], X[:,1], c=y)
plt.plot(x1, y1, '-', color='red', label='Linha do perceptron')
plt.title("Linha do perceptron")
plt.legend()
plt.show()
