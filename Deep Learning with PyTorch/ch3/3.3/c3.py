import torch

a = torch.tensor([1, 2, 3])
ua = a.unsqueeze(-1)
uua = ua.unsqueeze(0)
uua1 = ua.unsqueeze(-1)

print(f"a = {a} a.shape = {a.shape}")
print(f"ua = {ua} ua.shape = {ua.shape}")
print(f"uua = {uua} uua.shape = {uua.shape}")
print(f"uua1 = {uua1} uua1.shape = {uua1.shape}")

b = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
print(f"b = {b} b.shape = {b.shape}")
uua1Xb = b * uua1
print(f"uua1Xb = {uua1Xb}, uua1Xb.shape = {uua1Xb.shape}")


sum0 = uua1Xb.sum(0)
sum1 = uua1Xb.sum(1)
sum2 = uua1Xb.sum(2)
print(f"sum0 = {sum0} sum0.shape = {sum0.shape}")
print(f"sum1 = {sum1} sum1.shape = {sum1.shape}")
print(f"sum2 = {sum2} sum2.shape = {sum2.shape}")
