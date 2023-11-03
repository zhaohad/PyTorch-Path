import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5., 5., 0.1)
# exp 函数输入实数数组，返回每个元素 e^x 的值的数组
y = np.exp(x)

plt.plot(x, y, linewidth=2.0)
plt.title("numpy.exp")
plt.show()

x = np.arange(0.1, 5., 0.1)
# log 函数输入实数数组，返回每个元素 ln(x) 的值的数组
y = np.log(x)
plt.plot(x, y, linewidth=2.0)
plt.title("numpy.log")
plt.show()


np_array = np.array([1, 2, 3, 4])
# reshape 函数重新定义维度，新维度的元素个数与原维度元素个数必须保持一致
re_shape = np_array.reshape((2, 2))
print(f"np_array = {np_array}")
print(f"re_shape = {re_shape}")
