import torch
import torch.nn as nn

seq_model = nn.Sequential(nn.Linear(1, 13), nn.Tanh(), nn.Linear(13, 1))

print(f"seq_model = {seq_model}")

for param in seq_model.parameters():
    print(param.shape)

for name, param in seq_model.named_parameters():
    print(name, param.shape)
