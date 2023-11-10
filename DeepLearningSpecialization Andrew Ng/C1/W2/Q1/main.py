import numpy as np
from PIL import Image

img = Image.open("../../../Data/cat1.png")
img.show()

x_WHC = np.array(img)
print(f"x_HWC.shape = {x_WHC.shape}")

x_CWH = np.array(img).transpose(2, 1, 0)  # C -> W -> H
print(f"x_CWH.shape = {x_CWH.shape}")

c, w, h = x_CWH.shape
x_1 = x_CWH.reshape(c * w * h, 1)
print(f"x_1.shape = {x_1.shape}")

i1 = 0
i2 = 200
i3 = 492
index = (i1 * w + i2) * h + i3

print(f"x_CWH[{i1}][{i2}][{i3}] = {x_CWH[i1][i2][i3]}")
print(f"x_1[{i1}][{i2}][{i3}] = {x_1[index]}")
