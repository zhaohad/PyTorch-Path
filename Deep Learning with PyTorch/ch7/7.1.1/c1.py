from torchvision import datasets

data_path = "./dataset"

cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

print(f"cifar10 = {cifar10}, cifar10_val = {cifar10_val}, type(cifar10).__mro__ = {type(cifar10).__mro__}, len(cifar10) = {len(cifar10_val)}")
