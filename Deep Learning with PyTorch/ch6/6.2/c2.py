import torch.nn as nn
from collections import OrderedDict

seq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 8)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(8, 1))
]))

print(seq_model)

for param in seq_model.parameters():
    print(param.shape)

for name, param in seq_model.named_parameters():
    print(name, param.shape)

print(seq_model.output_linear.bias)
