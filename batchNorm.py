import numpy as np
from deepPy.layer import Layer


class BatchNorm(Layer):
    def __init__(self, input_shape=None, momentum=0.99, epsilon=1e-3, dtype="float32"):
        super().__init__()
        self.trainable = True
        self.input_shape = input_shape
        self.output_shape = None
        self.momentum = momentum
        self.epsilon = epsilon
        self.dtype = dtype

        # Trainable Parameters
        self.mean = None
        self.variance = None

        # For inference only
        self.moving_mean = None
        self.moving_variance = None

        # Storing normalized output for gradient calculations
        self.x_norm = None

    def initialize(self):
        self.output_shape = self.input_shape

        self.mean = np.zeros(self.input_shape)
        self.variance = np.ones(self.input_shape)

        self.moving_mean = np.zeros(self.input_shape)
        self.moving_variance = np.ones(self.input_shape)

    def forward(self, X):
        self.input = X

        self.batch_mean = np.mean(self.input, axis=0)
        self.batch_variance = 1. / np.sqrt(np.var(self.input, axis=0) + self.epsilon)

        # Calculating moving average
        self.moving_mean = self.moving_mean * self.momentum + (1 - self.momentum) *  self.batch_mean
        self.moving_variance = self.moving_variance * self.momentum + (1 - self.momentum) *  self.batch_variance

        # TODO: Implement alternative forward for predicting that uses the moving mean and variance
        self.x_norm = (self.input - self.batch_mean) * self.batch_variance
        self.output = (self.variance * self.x_norm) + self.mean
        return self.output

    def backward(self, output_gradient, batch_size):
        mean_gradient = np.sum(output_gradient, axis=0)
        variance_gradient = np.sum((output_gradient * self.x_norm), axis=0)

        output_gradient *= self.variance

        # Just for computing input gradient
        variance_delta = np.sum(output_gradient * (self.input - self.batch_mean), axis=0) * (-0.5 * self.batch_variance ** 3)
        # TODO: Maybe this --> mean_delta = np.sum(output_gradient * (-self.batch_variance), axis=0) + (variance_delta * np.mean(-2 * self.input - self.batch_mean, axis=0))
        mean_delta = np.sum(output_gradient * (-self.batch_variance), axis=0)

        # 1 / number of inputs in each channel
        # TODO: Not sure which of this is rights --> invN = 1. / np.prod(self.batch_mean.shape) or invN = 1. / self.batch_mean.shape[0]
        invN = 1. / self.batch_mean.shape[0]

        # Computing input gradient
        input_gradient = output_gradient * self.batch_variance + variance_delta * 2 * (self.input - self.batch_mean) * invN + mean_delta * invN

        # Updating Parameters
        self.variance, self.mean = self.optimizer.update((self.variance, self.mean), (variance_gradient, mean_gradient))

        return input_gradient
