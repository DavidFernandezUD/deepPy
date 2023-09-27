import numpy as np

def mse(y_pred, y_true, deriv=False):
    if deriv:
        return 2 * (y_pred - y_true)
    # Returns mean of all the samples
    return np.mean(np.square(y_pred - y_true))

def crossEntropy(y_pred, y_true, deriv=False):
    # TODO: Only works in combination with softmax, for now...
    y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
    if deriv:
        return y_pred - y_true
    correct_probabilities = np.sum(y_pred_clipped * y_true, axis=1, keepdims=True)
    log_probabilities = -np.log(correct_probabilities)
    # returns mean of all the samples
    return np.mean(log_probabilities)
