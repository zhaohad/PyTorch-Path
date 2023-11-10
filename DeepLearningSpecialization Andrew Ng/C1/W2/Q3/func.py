import numpy
import numpy as np


def linear(w, b, x):
    return w.T.dot(x) + b


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss(yhat, y):
    return -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
