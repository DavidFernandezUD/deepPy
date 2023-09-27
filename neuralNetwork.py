import deepPy.dense as dense
import deepPy.convolution as convolution
import deepPy.flatten as flatten
import deepPy.pooling as pooling
import deepPy.loss_functions as loss_functions
import deepPy.optimizers as optimizers
import deepPy.initializers as initializers
import deepPy.activations as activations
from copy import deepcopy
import numpy as np
import pickle
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self):
        self.loss = None
        self.optimizer = None
        self.initializer = None
        self.metrics = None

        self.layers = []
        self.batch_size = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss=loss_functions.mse, optimizer=optimizers.SGD(), initializer=initializers.init_random, metrics=("accuracy",)):
        self.loss = loss
        self.optimizer = optimizer
        self.initializer = initializer
        self.metrics = metrics

        if len(self.metrics) > 0:
            if "accuracy" in self.metrics:
                self.accuracy_history = []
                self.validation_accuracy = []
            if "loss" in self.metrics:
                self.loss_history = []
                self.validation_loss = []


        for i, layer in enumerate(self.layers):
            # For all but first layer
            if i > 0:
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
                layer.init = self.initializer
                layer.initialize()

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def fit(self, X, Y, epoch=100, batch_size=64, plot=False, validation=()):
        for epoch in range(epoch):
            shuffled_indexes = np.random.permutation(X.shape[0])
            X_shuffled = X[shuffled_indexes]
            Y_shuffled = Y[shuffled_indexes]

            # Accuracy and loss is calculated per batch for efficiency
            epoch_accuracy = []
            epoch_loss = []

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                Y_batch = Y_shuffled[i : i + batch_size]
                self.backward(X_batch, Y_batch, batch_size)

                if "accuracy" in self.metrics:
                    epoch_accuracy.append(self.accuracy(None, Y_batch))
                if "loss"in self.metrics:
                    epoch_loss.append(self.evaluate(None, Y_batch))

            # calculating metrics and plotting
            print(f"epoch {(epoch + 1):>3} ", end="")
            if len(self.metrics) > 0:
                if "loss" in self.metrics:
                    self.loss_history.append(np.mean(epoch_loss))
                    print(f"- loss={self.loss_history[-1]:.4f} ", end="")
                    if len(validation) == 2:
                        self.validation_loss.append(self.evaluate(*validation))
                        print(f"- val_loss={self.validation_loss[-1]:.4f} ", end="")
                if "accuracy" in self.metrics:
                    self.accuracy_history.append(np.mean(epoch_accuracy))
                    print(f"- accuracy={self.accuracy_history[-1]:.4f} ", end="")
                    if len(validation) == 2:
                        self.validation_accuracy.append(self.accuracy(*validation))
                        print(f"- val_accuracy={self.validation_accuracy[-1]:.4f} ", end="")
                print()

                # Plotting Metrics
                # TODO: Add subplots for booth loss and accuracy metrics
                if plot:
                    plt.cla()
                    plt.plot(self.validation_accuracy, c="orange", label="validation")
                    plt.plot(self.accuracy_history, c="skyblue", label="training")
                    plt.xlabel("Epoch")
                    plt.ylabel("Accuracy")
                    plt.title("Model Accuracy")
                    plt.legend()
                    plt.show(block=False)
                    plt.pause(0.001)

    def backward(self, X, Y, batch_size):
        output_gradient = self.loss(self.predict(X), Y, deriv=True)
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, batch_size)

    def evaluate(self, X, Y):
        if X is None:
            return self.loss(self.layers[-1].output, Y)
        else:
            return self.loss(self.predict(X), Y)

    def accuracy(self, X, Y):
        if X is None:
            # Reuses the already calculated output of the network instead of calculating it again
            predictions = np.argmax(self.layers[-1].output, axis=1)
        else:
            predictions = np.argmax(self.predict(X), axis=1)
        real = np.argmax(Y, axis=1)
        accuracy = np.mean(predictions == real)
        return accuracy

    def save_model(self, filepath):
        with open(filepath, 'wb') as fp:
            pickle.dump(self.__dict__, fp, 2)

    def load_model(self, filepath):
        with open(filepath, 'rb') as fp:
            tmp_dict = pickle.load(fp)

        self.__dict__.clear()
        self.__dict__.update(tmp_dict)

    def save_state(self):
        # TODO: save just the state of the weights and biases (no layer types)
        pass

    def load_state(self):
        # TODO: load weights and bias state from file (not reconstructing the layers from 0)
        pass

    # TODO: Add optimizer and initializer info
    def info(self):
        params = 0
        if isinstance(self.layers[0], convolution.Conv2D):
            layers = f"INPUT{self.layers[0].input_shape} -> "
        else:
            layers = f"INPUT({self.layers[0].weights.shape[0]}) -> "
        loss = self.loss.__name__
        for i, layer in enumerate(self.layers):
            if isinstance(layer, dense.Dense):
                params += layer.weights.size
                params += layer.bias.size
                layers += f"Dense{layer.weights.shape} -> "
            elif isinstance(layer, convolution.Conv2D):
                params += layer.kernels.size
                params += layer.bias.size
                layers += f"Conv{layer.depth, (layer.kernel_size, layer.kernel_size)} -> "
            else:
                layers += f"{layer.__class__.__name__}() -> "
        layers += f"OUTPUT({self.layers[-2].weights.shape[1]})"

        print(f"layers => {layers}\nparams => {params}\nloss => {loss}")

    # TODO: Update this method
    def performance(self, filepath=None):
        if self.metrics:
            plt.cla()
            plt.plot(self.validation_accuracy, c="orange", label="validation")
            plt.plot(self.accuracy_history, c="skyblue", label="training")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Model Accuracy")
            if filepath is not None:
                plt.savefig(filepath)
            plt.show()
        else:
            print("No Metrics Available")
