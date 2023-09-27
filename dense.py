from deepPy.layer import Layer
import numpy as np


class Dense(Layer):
    def __init__(self, output_shape, input_shape=None, dtype="float32"):
        super().__init__()
        self.trainable = True
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dtype = dtype
        self.weights = None
        self.bias = None

    def initialize(self):
        self.init(self, self.dtype)

    def forward(self ,X):
        self.input = X
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient, batch_size):
        weights_gradient = np.dot(self.input.T, output_gradient) / batch_size
        bias_gradient = np.mean(output_gradient, axis=0)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights, self.bias = self.optimizer.update((self.weights, self.bias), (weights_gradient, bias_gradient))

        return input_gradient
