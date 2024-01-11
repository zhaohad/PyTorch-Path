# https://www.bilibili.com/video/BV1vU4y1A7QJ/?p=3&spm_id_from=pageDriver&vd_source=6d7e1195832f8e47dd489bced2b238a7
import torch.utils.data
from torchvision import datasets, transforms

from LeNet import LeNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

data_path = "../../data"
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root=data_path, transform=transform, train=True, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

test_dataset = datasets.MNIST(root=data_path, transform=transform, train=False, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

net = LeNet().to(DEVICE)

# LOSS
loss_fn = torch.nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

# 学习率每建个10轮，变为原来的0.1
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def train(dataloader, net, loss_fn, optimizer):
    loss_p, n = 0.0, 0
    for batch, (X, y) in enumerate(dataloader):
        # batch: batch number
        # X: 每个图片
        # y: 图片label
        X, y = X.to(DEVICE), y.to(DEVICE)
        output = net(X)
        cur_loss = loss_fn(output, y)

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss_p += cur_loss.item()
        n += 1
    loss = loss_p / n
    print(f"loss: {loss}")


def val(dataloader, net):
    acc_p, n = 0.0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            output = net(X)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            acc_p += cur_acc.item()
            n += 1
    accuracy = acc_p / n
    print(f"accuracy: {accuracy}")


if __name__ == "__main__":
    EPOCH_MAX = 1000
    for epoch in range(EPOCH_MAX):
        print(f"Epoch: {epoch}")
        for param_group in optimizer.param_groups:
            print(f"lr: {param_group['lr']}")
        train(train_dataloader, net, loss_fn, optimizer)
        val(test_dataloader, net)

    torch.save(net.state_dict(), "model.pth")
