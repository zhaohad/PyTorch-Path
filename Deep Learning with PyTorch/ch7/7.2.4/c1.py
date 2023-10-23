import torch
from torch import nn


def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()

t = torch.tensor([1.0, 2.0, 3.0])

print(f"softmax(t) = {softmax(t)}")

softmax = nn.Softmax()
t = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

print(f"nn.Softmax(t) = {softmax(t)}")
