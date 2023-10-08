from torchvision import models
from torchvision import transforms
from PIL import Image
import torch

# print("before dir")
ms = dir(models)

# print(f"ms = {ms}")

alexnet = models.AlexNet()
resnet = models.resnet101(pretrained = True)

# print(f'alexnet = {alexnet}')

# print(f'resnet = {resnet}')

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )])

# print(f"preprocess = {preprocess}")

img = Image.open("./phone2.jpg")
print(f'img = {img}')

# img.show()

img_t = preprocess(img)

# print(f'img_t = {img_t}')

batch_t = torch.unsqueeze(img_t, 0)
# print(f"batch_t = {batch_t}")

resnet.eval()

out = resnet(batch_t)

# print(out)

with open("./imagenet_classes.txt") as f:
    labels = [line.strip() for line in f]

a, index = torch.max(out, 1)
# print(f'a = {a}')
# print(f'index = {index[0]}')

percentage = torch.nn.functional.softmax(out, dim = 1)[0] * 100
# print(f"lables = {labels}")
label = labels[index[0]]
print(f'label = {label} percentage = {percentage[index[0]].item()}')

