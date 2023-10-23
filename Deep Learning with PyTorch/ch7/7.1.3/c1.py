from torchvision import datasets
from matplotlib import pyplot as plt
from torchvision import transforms

data_path = "../dataset"

cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

img, label = cifar10[99]

print(f"dir(transforms) = {dir(transforms)}")

to_tensor = transforms.ToTensor()
img_t = to_tensor(img)

print(f"img_t.shape = {img_t.shape}")

tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=to_tensor)
img_t, _ = tensor_cifar10[99]
print(f"type(img_t) = {type(img_t)}")
print(f"img_t.shape = {img_t.shape}, img_t.dtype = {img_t.dtype}")
print(f"img_t.min() = {img_t.min()}, img_t.max() = {img_t.max()}")

plt.imshow(img_t.permute(1, 2, 0))  # pyplot 只接受H W C的顺序
plt.show()
