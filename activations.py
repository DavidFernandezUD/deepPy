from deepPy.layer import Layer
from deepPy.loss_functions import crossEntropy
import numpy as np


# Base Activation Class
class Activation(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        pass

    def backward(self, output_gradient, batch_size):
        pass

class Sigmoid(Activation):
    def __init__(self, input_shape=None):
        super().__init__()
        self.trainable = False
        self.input_shape = input_shape
        self.output_shape = None

    def forward(self, X):
        self.input = X
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backward(self, output_gradient, batch_size):
        input_gradient = (self.output * (1 - self.output)) * output_gradient
        return input_gradient


class Tanh(Activation):
    def __init__(self, input_shape=None):
        super().__init__()
        self.trainable = False
        self.input_shape = input_shape
        self.output_shape = None

    def forward(self, X):
        self.input = X
        exp2 = np.exp(2 * self.input)
        self.output = (exp2 - 1) / (exp2 + 1)
        return self.output

    def backward(self, output_gradient, batch_size):
        input_gradient = (1 - np.square(self.output)) * output_gradient
        return input_gradient


class ReLU(Activation):
    def __init__(self, input_shape=None):
        super().__init__()
        self.trainable = False
        self.input_shape = input_shape
        self.output_shape = None

    def forward(self, X):
        self.input = X
        self.output = np.maximum(0, self.input)
        return self.output

    def backward(self, output_gradient, batch_size):
        input_gradient = (np.where(self.input>0, 1, 0)) * output_gradient
        return input_gradient


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01, input_shape=None):
        super().__init__()
        self.trainable = False
        self.input_shape = input_shape
        self.output_shape = None
        self.alpha = alpha

    def forward(self, X):
        self.input = X
        self.output = np.maximum(self.alpha * self.input, self.input)
        return self.output

    def backward(self, output_gradient, batch_size):
        input_gradient = (np.where(self.input>0, 1, self.alpha)) * output_gradient
        return input_gradient


class PReLU(Activation):
    def __init__(self, input_shape=None, dtype="float32"):
        super().__init__()
        self.trainable = True
        self.input_shape = input_shape
        self.output_shape = None
        self.dtype = dtype
        self.alpha = None

    def initialize(self):
        self.init(self, self.dtype)

    def forward(self, X):
        self.input = X
        self.output = np.maximum(0, self.input) + np.minimum(0, self.input) * self.alpha
        return self.output

    def backward(self, output_gradient, batch_size):
        # Taking the mean of all the samples in the batch
        alpha_gradient = np.mean(np.where(self.input > 0, 0, output_gradient * self.input), axis=0)
        input_gradient = np.where(self.input > 0, output_gradient, output_gradient * self.alpha)

        self.alpha = self.optimizer.update(self.alpha, alpha_gradient)

        return input_gradient


class SiLU(Activation):
    def __init__(self, input_shape=None):
        super().__init__()
        self.trainable = False
        self.input_shape = input_shape
        self.output_shape = None

    def forward(self, X):
        self.input = X
        self.output = self.input / (1 + np.exp(-self.input))
        return self.output

    def backward(self, output_gradient, batch_size):
        exp = np.exp(-self.input)
        input_gradient = ((1 + exp + self.input * exp) / np.square(1 + exp)) * output_gradient
        return input_gradient


class ELU(Activation):
    def __init__(self, input_shape=None, alpha=0.1):
        super().__init__()
        self.trainable = False
        self.input_shape = input_shape
        self.output_shape = None
        self.alpha = alpha

    def forward(self, X):
        self.input = X
        self.output = np.where(self.input >= 0, self.input, self.alpha * (np.exp(self.input - 1)))
        return self.output

    def backward(self, output_gradient, batch_size):
        input_gradient = (np.where(self.input > 0, 1, self.alpha * np.exp(self.input))) * output_gradient
        return input_gradient


class Gaussian(Activation):
    def __init__(self, input_shape=None):
        super().__init__()
        self.trainable = False
        self.input_shape = input_shape
        self.output_shape = None

    def forward(self, X):
        self.input = X
        self.output = np.exp(-np.square(self.input))
        return self.output

    def backward(self, output_gradient, batch_size):
        input_gradient = (-2 * self.input * self.output) * output_gradient
        return input_gradient


class Softmax(Activation):
    def __init__(self, input_shape=None):
        super().__init__()
        self.trainable = False
        self.input_shape = input_shape
        self.output_shape = None
        self.crossEntropy = False

    def forward(self, X):
        self.input = X
        safe_values = self.input - np.max(self.input, axis=1)[:, np.newaxis]
        exp_values = np.exp(safe_values)
        output = exp_values / np.sum(exp_values, axis=1)[:, np.newaxis]
        self.output = output
        return self.output

    def backward(self, output_gradient, batch_size):
        # Bypassing the gradient calculation if crossEntropy is used
        if self.crossEntropy:
            return output_gradient
        input_size = self.output.shape[1]
        M = np.tile(np.mean(self.output, axis=0), (input_size, 1))
        input_gradient = np.dot(output_gradient, np.multiply(M, (np.identity(input_size) - M.T)))
        return input_gradient
