import numpy as np


class Optimizer:
    def __init__(self, learning_rate=1e-2, decay=0., lr_min=0.0, lr_max=np.inf):
        self.learning_rate = learning_rate
        self.decay = decay
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.iterations = 0

    def update(self, params, gradients):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate=1e-2, decay=0., lr_min=0.0, lr_max=np.inf):
        super().__init__(learning_rate, decay, lr_min, lr_max)

    def update(self, params, gradients):
        for p, g in zip(params, gradients):
            p -= g * self.learning_rate

        return params


class Momentum(Optimizer):
    def __init__(self, momentum=0.9, learning_rate=1e-2, decay=0., lr_min=0.0, lr_max=np.inf):
        super().__init__(learning_rate, decay, lr_min, lr_max)
        self.momentum = momentum
        self.velocity = None

    def update(self, params, gradients):
        if self.velocity is None:
            self.velocity = [np.zeros(shape=param.shape, dtype="float32") for param in params]

        for i, (v, p, g) in enumerate(zip(self.velocity, params, gradients)):
            v = self.momentum * v + (1 - self.momentum) * g
            p -= v * self.learning_rate
            self.velocity[i] = v

        return params


class RMSprop(Optimizer):
    def __init__(self, beta=0.9, epsilon=1e-7, learning_rate=1e-3, decay=0., lr_min=0.0, lr_max=np.inf):
        super().__init__(learning_rate, decay, lr_min, lr_max)
        self.beta = beta
        self.epsilon = epsilon
        self.exp_average = None

    def update(self, params, gradients):
        if self.exp_average is None:
            self.exp_average = [np.zeros(shape=param.shape, dtype="float32") for param in params]

        for i, (s, p, g) in enumerate(zip(self.exp_average, params, gradients)):
            s = self.beta * s + (1 - self.beta) * np.square(g)
            p -= (g / (np.sqrt(s + self.epsilon))) * self.learning_rate
            self.exp_average[i] = s

        return params


class ADAM(Optimizer):
    def __init__(self, beta_1=0.9, beta_2=0.999, epsilon=1e-7, learning_rate=1e-3, decay=0., lr_min=0.0, lr_max=np.inf):
        super().__init__(learning_rate, decay, lr_min, lr_max)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.velocity = None
        self.exp_average = None

    def update(self, params, gradients):
        if self.exp_average is None:
            self.velocity = [np.zeros(shape=param.shape, dtype="float32") for param in params]
            self.exp_average = [np.zeros(shape=param.shape, dtype="float32") for param in params]

        for i, (v, s, p, g) in enumerate(zip(self.velocity, self.exp_average, params, gradients)):
            v = self.beta_1 * v + (1 - self.beta_1) * g
            s = self.beta_2 * s + (1 - self.beta_2) * g * g
            p -= (v / (np.sqrt(s) + self.epsilon)) * self.learning_rate
            self.velocity[i] = v
            self.exp_average[i] = s

        return params
