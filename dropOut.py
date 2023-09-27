import numpy as np
from deepPy.activations import Activation


# Useful for avoiding overfitting
# TODO: Disable dropout when predicting (Should only work when training)
class DropOut(Activation):
    def __init__(self, prob=0.1, input_shape=None):
        super().__init__()
        self.trainable = False
        self.input_shape = input_shape
        self.output_shape = None
        self.prob = prob
        self.rand_values = None

        # To compensate for the dropout loss
        self.scale_up = 1 / (1 - self.prob)

    def forward(self, X):
        self.input = X
        self.rand_values = np.where(np.random.uniform(low=0., high=1., size=self.input_shape) > self.prob, 1, 0)
        self.output = self.input * self.rand_values * self.scale_up
        return self.output

    def backward(self, output_gradient, batch_size):
        input_gradient = self.rand_values * output_gradient * self.scale_up
        return input_gradient