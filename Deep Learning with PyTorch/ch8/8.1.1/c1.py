import torch.nn as nn

conv = nn.Conv2d(3, 16, kernel_size=3)

print(f"conv = {conv}")

print(f"conv.weight.shape = {conv.weight.shape}, conv.bias.shape = {conv.bias.shape}")
