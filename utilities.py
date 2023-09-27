import numpy as np

def to_one_hot(Y):
    n_categories = np.max(Y) + 1
    y_encoded = []
    for data in Y:
        one_hot = [1 if i == data else 0 for i in range(n_categories)]
        y_encoded.append(one_hot)
    return np.array(y_encoded)
