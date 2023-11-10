import numpy as np
from PIL import Image
import func

img = Image.open("../../../Data/cat1.png")
y = 1
img.show()

x_cwh = np.array(img, dtype=np.float32).transpose(2, 1, 0)
print(f"x_cwh.shape = {x_cwh.shape} x_cwh = {x_cwh}")

channel, width, height = x_cwh.shape
len_feature = channel * width * height
x = x_cwh.reshape(len_feature, 1)
x = (x - 128) / 128  # 归一化，将颜色数值限制在-1 ~ 1

print(f"x_1.shape = {x.shape} x = {x}")

w = np.random.randn(len_feature, 1)
# w = np.zeros((len_feature, 1))
w = w / w.max()
b = np.random.randn(1, 1)

print(f"w = {w} b = {b}")

z = func.linear(w, b, x)
a = func.sigmoid(z)
print(f"z = {z} a = {a}")
print(a)
loss = func.loss(a, y)
print(f"loss = {loss}")
