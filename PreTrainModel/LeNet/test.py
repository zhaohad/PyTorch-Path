import torch.utils.data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from LeNet import LeNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

data_path = "../../data"
transform = transforms.Compose([transforms.ToTensor()])

test_dataset = datasets.MNIST(root=data_path, transform=transform, train=True, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

net = LeNet().to(DEVICE)
net.load_state_dict(torch.load("model.pth"))
net.eval()

data_iter = iter(test_dataloader)
datas, clss = next(data_iter)
datas = datas.to(DEVICE)
clss = clss.to(DEVICE)

print(f"clss.shape = {clss.shape}")
print(f"datas.shape = {datas.shape}")

with torch.no_grad():
    feat = net(datas)
    _, pred = torch.max(feat, axis=1)
    print(f"pred.shape = {pred.shape}")
    print(f"feat.shape = {feat.shape}")
    print(pred)
    for i in range(datas.shape[0]):
        data = datas[i]
        img = transforms.ToPILImage()(data)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
