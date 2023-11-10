# numpy构建基本函数

1. 读取图片
2. 图片转换为Numpy数组
3. 转换数组维度为c, w, h
4. 转为列向量
5. 按列向量读取某索引数据

# Answer
1. 读取图片:

```python
import numpy as np
from PIL import Image

img = Image.open("../../../Data/cat1.png")
img.show()
```
2. 图片转换为Numpy数组:
```python
x_WHC = np.array(img)
print(f"x_HWC.shape = {x_WHC.shape}")
```
3. 转换数组维度为c, w, h:
```python
x_CWH = np.array(img).transpose(2, 1, 0)  # C -> W -> H
print(f"x_CWH.shape = {x_CWH.shape}")
```
4. 转为列向量:
```python
c, w, h = x_CWH.shape
x_1 = x_CWH.reshape(c * w * h, 1)
print(f"x_1.shape = {x_1.shape}")
```
5. 按列向量读取某索引数据
```python
i1 = 0
i2 = 200
i3 = 492
index = (i1 * w + i2) * h + i3

print(f"x_CWH[{i1}][{i2}][{i3}] = {x_CWH[i1][i2][i3]}")
print(f"x_1[{i1}][{i2}][{i3}] = {x_1[index]}")
```