from torchvision import datasets
from matplotlib import pyplot as plt
from torchvision import transforms
import torch

data_path = "../dataset"

cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

img, label = cifar10[99]

to_tensor = transforms.ToTensor()
img_t = to_tensor(img)

tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=to_tensor)
img_t, _ = tensor_cifar10[99]

imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3)
print(f"imgs.shape = {imgs.shape}")
imgs_mean = imgs.view(3, -1).mean(dim=1)
imgs_std = imgs.view(3, -1).std(dim=1)
imgs_norm = transforms.Normalize(imgs_mean, imgs_std)  # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.4914, 0.4822, 0.4465))
print(f"imgs_mean = {imgs_mean}, imgs_std = {imgs_std}, imgs_norm = {imgs_norm}")

transformed_cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.Compose([to_tensor, imgs_norm]))
img_t, _ = transformed_cifar10[99]
plt.imshow(img_t.permute(1, 2, 0))
plt.show()
