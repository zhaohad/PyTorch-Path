# numpy构建基本函数

1. 画出sigmoid函数
2. 画出sigmoid的梯度函数
3. 画出ReLU函数

# Answer
1. sigmoid函数： S(x) = 1 / (1 + e ^ (-x))

![alt 属性文本](sigmoid_function.png "sigmoid_function()")

![alt 属性文本](sigmoid.png "sigmoid()")

2. sigmoid的梯度函数：

![alt 属性文本](sigmoid_derivative_function.jpg "sigmoid_derivative_function()")

![alt 属性文本](sigmoid_derivative.png "sigmoid_derivative()")

3. ReLU函数：y = max(0, x)

![alt 属性文本](ReLU.png "rectified_linear_unit()")



```python
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


```

