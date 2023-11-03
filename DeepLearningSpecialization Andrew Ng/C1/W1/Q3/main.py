import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def rectified_linear_unit(x):
    return np.maximum(0, x)

x = np.arange(-10., 10., 0.1)
y = sigmoid(x)
plt.plot(x, y, linewidth=2.0)
plt.title("sigmoid")
plt.show()

y = sigmoid_derivative(x)
plt.plot(x, y, linewidth=2.0)
plt.title("sigmoid_derivative")
plt.show()

y = rectified_linear_unit(x)
plt.plot(x, y, linewidth=2.0)
plt.title("Rectified Linear Unit")
plt.show()
