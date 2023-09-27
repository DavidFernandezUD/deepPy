from deepPy.layer import Layer
import numpy as np


class Flatten(Layer):
    def __init__(self, input_shape=None):
        super().__init__()
        self.trainable = False
        self.input_shape = input_shape
        self.output_shape = None

    # Updating the layer when adding it to the model
    def initialize(self):
        self.output_shape = (self.input_shape[0] * self.input_shape[1] * self.input_shape[2])

    def forward(self, X):
        self.input = X
        self.output = np.reshape(self.input, (self.input.shape[0], self.output_shape))
        return self.output

    def backward(self, output_gradient, batch_size):
        return np.reshape(output_gradient, (self.input.shape[0], *self.input_shape))