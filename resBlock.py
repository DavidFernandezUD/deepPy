from deepPy.layer import Layer
import deepPy.flatten as flatten
import deepPy.pooling as pooling
import deepPy.loss_functions as loss_functions
import deepPy.optimizers as optimizers
import deepPy.initializers as initializers
import deepPy.activations as activations
from copy import deepcopy
import numpy as np


class ResBlock(Layer):
    def __init__(self):
        self.trainable = True
        self.optimizer = None
        self.initializer = None
        self.layers = []

        self.input_shape = None
        self.output_shape = None

    def add(self, layer):
        self.layers.append(layer)

    # Initialize method in the ResBlock class also compiles its internal layers
    def initialize(self, optimizer=optimizers.SGD(), initializer=initializers.init_random):
        self.optimizer = optimizer
        self.init = initializer

        # Input and output shapes must be equal to perform the sum with the shortcut
        if self.layers[0].input_shape is not None:
            self.input_shape = self.layers[0].input_shape
        self.output_shape = self.input_shape

        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.input_shape = self.input_shape
            else:
                layer.input_shape = self.layers[i - 1].output_shape


            # Assigning output_shape automatically to activation layers
            if isinstance(layer, activations.Activation):
                layer.output_shape = layer.input_shape
                # Checking for cross entropy and softmax
                if isinstance(layer, activations.Softmax) and self.loss == loss_functions.crossEntropy:
                    layer.crossEntropy = True

            # Calculating output_shapes for special layers
            if isinstance(layer, flatten.Flatten) or isinstance(layer, pooling.MaxPooling):
                layer.initialize()

            if layer.trainable:
                layer.optimizer = deepcopy(self.optimizer)
                layer.init = self.init
                layer.initialize()

    def forward(self, X):
        self.input = np.copy(X)
        for layer in self.layers:
            X = layer.forward(X)

        # The output recombines with the shortcut
        self.output = X + self.input
        return self.output

    def backward(self, output_gradient, batch_size):
        output_gradient = output_gradient
        input_gradient = np.copy(output_gradient)
        for layer in reversed(self.layers):
            input_gradient = layer.backward(input_gradient, batch_size)

        # Gradient also recombines with the skip connection
        return input_gradient + output_gradient
