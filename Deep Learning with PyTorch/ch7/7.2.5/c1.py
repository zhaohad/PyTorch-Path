import torch
from torch import nn
from torchvision import datasets
from matplotlib import pyplot as plt
from torchvision import transforms

data_path = "../dataset"

to_tensor = transforms.ToTensor()
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]

model = nn.Sequential(nn.Linear(3072, 512), nn.Tanh(), nn.Linear(512, 2), nn.LogSoftmax(dim=1))
loss = nn.NLLLoss()


img, label = cifar2[0]
img = to_tensor(img)

out = model(img.view(-1).unsqueeze(0))
res = loss(out, torch.tensor([label]))

print(f"out = {out}, res = {res}")
