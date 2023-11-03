# numpy构建基本函数

1. Numpy是Python中主要的科学计算包。它由一个大型社区维护。
2. 在本练习中，你将学习一些关键的numpy函数，例如np.exp，np.log和np.reshape。
3. 你需要知道如何使用这些函数去完成将来的练习。

# Answer
1. exp 函数输入实数数组，返回每个元素 e^x 的值的数组
![alt 属性文本](np_exp.png "numpy.exp()")
2. log 函数输入实数数组，返回每个元素 ln(x) 的值的数组
![alt 属性文本](np_log.png "numpy.log()")
3. reshape 函数重新定义维度，新维度的元素个数与原维度元素个数必须保持一致
python code:
``` python
np_array = np.array([1, 2, 3, 4])
# reshape 函数重新定义维度，新维度的元素个数与原维度元素个数必须保持一致
re_shape = np_array.reshape((2, 2))
print(f"np_array = {np_array}")
print(f"re_shape = {re_shape}")
```

output:
```text
np_array = [1 2 3 4]
re_shape = [[1 2]
 [3 4]]
```