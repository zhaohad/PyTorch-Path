import torch
import numpy as np
import sys

sys.path.append("../..")

from hwutil.utils import dump

x = torch.ones(())
y = torch.ones(3, 1)
z = torch.ones(1, 3)
a = torch.ones(2, 1, 1)


xy = x * y
yz = y * z
yza = y * z * a

dump(xy)
dump(yz)
dump(yza)

# 创建两个张量
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 在新的维度上堆叠这两个张量
stacked_tensor = torch.stack((a, b), dim=1)

dump(stacked_tensor)
