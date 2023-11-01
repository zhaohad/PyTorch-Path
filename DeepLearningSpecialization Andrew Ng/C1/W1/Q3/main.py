import numpy as np

py_array = [1, 2, 3, 4]
np_array = np.array([1, 2, 3, 4])

print(np.__version__)

print(f"py_array = {py_array}, np_array = {np_array}, np_array.shape = {np_array.shape}")

# exp 函数输入实数数组，返回每个元素 e^x 的值的数组
print(f"np.exp(np_array) = {np.exp(np_array)}")

# log 函数输入实数数组，返回每个元素 ln(x) 的值的数组
print(f"np.log(np_array) = {np.log(np_array)}")

# reshape 函数重新定义维度，新维度的元素个数与原维度元素个数必须保持一致
re_shape = np_array.reshape((2, 2))
print(f"np_array = {np_array}, re_shape = {re_shape}")

