
class Layer:
    def __init__(self):
        self.trainable = False
        self.input_shape = None
        self.output_shape = None
        self.input = None
        self.output = None
        self.optimizer = None
        self.init = None

    def forward(self, X):
        pass

    def backward(self, output_gradient, batch_size):
        pass
