from .layer import Layer
import numpy as np


class MaxPooling(Layer):
    def __init__(self, pool_size, input_shape=(1, 1, 1)):
        super().__init__()
        self.trainable = False
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.output_shape = None

    # Updating the layer when compiling the model
    def initialize(self):
        self.output_shape = (self.input_shape[0], (self.input_shape[1] // self.pool_size), (self.input_shape[2] // self.pool_size))

    def forward(self, X):
        self.input = X
        self.output = np.zeros((self.input.shape[0], *self.output_shape))

        # Precalculating input gradient in forward pass for optimization
        self.input_gradient = np.zeros((self.input.shape[0], *self.input_shape))

        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                pool = X[:, :, i*self.pool_size:(i + 1)*self.pool_size, j*self.pool_size:(j + 1)*self.pool_size]
                maximum = np.max(pool, axis=(2, 3))
                self.output[:, :, i, j] = maximum

                # This creates a matrix with ones on the max values
                pool_max = np.where(pool == np.max(pool, axis=(2, 3), keepdims=True), 1, 0)
                self.input_gradient[:, :, i * self.pool_size:(i + 1) * self.pool_size,j * self.pool_size:(j + 1) * self.pool_size] = pool_max

        return self.output

    def backward(self, output_gradient, batch_size):
        output_gradient = np.kron(output_gradient, np.ones((self.pool_size, self.pool_size)))

        vertical_padding = self.input_shape[1] - output_gradient.shape[2]
        horizontal_padding = self.input_shape[2] - output_gradient.shape[3]

        output_gradient = np.pad(output_gradient, ((0, 0), (0, 0), (0, vertical_padding), (0, horizontal_padding)))

        self.input_gradient *= output_gradient
        return self.input_gradient
