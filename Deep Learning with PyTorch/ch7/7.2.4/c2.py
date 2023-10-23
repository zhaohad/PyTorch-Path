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

model = nn.Sequential(nn.Linear(3072, 512), nn.Tanh(), nn.Linear(512, 2), nn.Softmax(dim=1))


img, _ = cifar2[0]
img = to_tensor(img)

img_batch = img.view(-1).unsqueeze(0)

plt.imshow(img.permute(1, 2, 0))
plt.show()
