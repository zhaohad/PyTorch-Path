# https://blog.csdn.net/Aaron_neil/article/details/130168355

import torch

# 创建一个3x4的张量
tensor = torch.zeros(3, 4).long()
print(f"原始张量: shape = {tensor.shape}")
print(tensor)

# 要更新的索引位置
index = torch.tensor([[0, 1, 2, 1],   # 行索引
                      [1, 2, 0, 2]])  # 列索引

print(f"index.shape = {index.shape}")

# 新值
v = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

# 在指定位置上使用scatter_()方法更新张量
tensor.scatter_(0, index, v)

print(f"更新后的张量: shape = {tensor.shape}")
print(tensor)

x = torch.rand(2, 5)

x = torch.tensor([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]])
y = torch.zeros(3, 5).long().scatter_(0, torch.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)

print(f"y = {y}")

