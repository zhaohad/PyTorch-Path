from torchvision import datasets
from matplotlib import pyplot as plt


data_path = "./dataset"

cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

print(f"cifar10 = {cifar10}\ncifar10_val = {cifar10_val}\ntype(cifar10).__mro__ = {type(cifar10).__mro__}\nlen(cifar10) = {len(cifar10_val)}")

img, label = cifar10[99]

print(f"img = {img}, label = {label}")

plt.imshow(img)
plt.show()
