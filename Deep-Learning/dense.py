import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, n_neurons, n_inputs) -> None:
        self.weights = np.random.randn(n_neurons, n_inputs)
        self.biases = np.ones(n_neurons)
    def forward(self, x):
        self.input = x
        self.output = np.dot(self.weights, x) + self.biases
        return self.output
    def backward(self, output_gradient, learning_Rate):
        weights_gradient = np.dot(output_gradient, self.input)
        self.weights -= weights_gradient * learning_Rate
        self.biases -= output_gradient * learning_Rate
        return np.dot(self.weights, output_gradient)