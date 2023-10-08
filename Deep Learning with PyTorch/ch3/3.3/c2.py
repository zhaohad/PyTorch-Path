import torch

img_t = torch.randn(3, 5, 5)

print(f"img_t.shape = {img_t.shape}, img_t = {img_t}")

x = torch.tensor([[1.0, 2.0, 4], [4.0, 5.0, 6.0]])

print(f"x = {x}")

meanx = torch.mean(x, dim = 0, keepdim = False)

meanx1 = torch.mean(x)

meanx2 = torch.mean(x, dim = 1, keepdim = True)

print(f"meanx = {meanx}, meanx.shape = {meanx.shape}\n meanx1 = {meanx1}, meanx1.shape = {meanx1.shape}")
print(f"meanx = {meanx2}, meanx2.shape = {meanx2.shape}")

mean3 = x.mean(-1)
print(f"mean3 = {mean3} mean3.shape = {mean3.shape}")
