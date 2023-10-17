import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-6, 6, 100)  # 生成从-6到6的100个点
y = sigmoid(x)

plt.figure(figsize=(6, 6))
plt.plot(x, y, label='Sigmoid Function', color='b')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.title('Sigmoid Function')
plt.grid(True)
plt.legend()
plt.show()
