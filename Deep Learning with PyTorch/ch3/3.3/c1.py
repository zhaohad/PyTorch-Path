import torch

points = torch.zeros(3)

print(f"points = {points}")
print(f"points[None] = {points[None]}")

_2dp = torch.tensor([[1, 2], [3, 4]])

print(f"_2dp = {_2dp}")
print(f"_2dp[None] = {_2dp[None]}")
