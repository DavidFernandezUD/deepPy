from deepPy.layer import Layer
import numpy as np
from scipy import signal


class Conv2D(Layer):
    def __init__(self, depth, kernel_size, input_shape=None, padding="valid", dtype="float32"):
        super().__init__()
        self.trainable = True
        self.depth = depth
        self.kernel_size = kernel_size
        self.input_shape = input_shape

        # When manually imputing the input_shape
        if input_shape is not None:
            self.input_depth = self.input_shape[0]
            self.input_height = self.input_shape[1]
            self.input_width = self.input_shape[2]

        self.padding = padding
        self.dtype = dtype

        # None Until Initialized
        self.kernel_shape = None
        self.output_shape = None

        self.kernels = None
        self.bias = None

    def initialize(self):
        # Updating new input_shape
        self.input_depth = self.input_shape[0]
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]

        self.kernel_shape = (self.depth, self.input_depth, self.kernel_size, self.kernel_size)

        if self.padding == "valid":
            self.output_shape = (self.depth, (self.input_height - self.kernel_size + 1), (self.input_width - self.kernel_size + 1))
        elif self.padding == "same":
            self.output_shape = (self.depth, self.input_height, self.input_width)
        self.init(self, self.dtype)

    def forward(self, X):
        self.input = X
        self.output = np.tile(self.bias, (self.input.shape[0], 1, 1, 1))

        for i in range(self.input.shape[0]):
            for j in range(self.depth):
                for k in range(self.input_depth):
                    self.output[i, j] += signal.correlate2d(self.input[i, k], self.kernels[j, k], mode=self.padding)

        return self.output

    def backward(self, output_gradient, batch_size):
        input_gradient = np.zeros((self.input.shape[0], *self.input_shape))
        kernels_gradient = np.zeros(self.kernel_shape)
        bias_gradient = np.mean(output_gradient, axis=0)

        for i in range(self.input.shape[0]):
            for j in range(self.depth):
                for k in range(self.input_depth):
                    kernels_gradient[j, k] += signal.correlate2d(self.input[i, k], output_gradient[i, j], mode="valid")
                    if self.padding == "same":
                        input_gradient[i, k] += signal.convolve2d(output_gradient[i, j], self.kernels[j, k],mode="same")
                    else:
                        input_gradient[i, k] += signal.convolve2d(output_gradient[i, j], self.kernels[j, k], mode="full")

        kernels_gradient /= batch_size # Normalizing for batch size

        self.kernels, self.bias = self.optimizer.update((self.kernels, self.bias), (kernels_gradient, bias_gradient))

        return input_gradient
