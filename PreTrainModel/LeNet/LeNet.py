import torch
from torch import nn


class LeNet(nn.Module):

    def __init__(self,):
        super(LeNet, self).__init__()

        self.Sigmoid = nn.Sigmoid()
        self.Flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)  # wh: 28 -> 28, c: 1 -> 6
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # wh: 28 -> 14, c: 6
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  # wh: 14 -> 10, c: 6 -> 16
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)  # wh: 10 -> 5, c: 16
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)  # wh: 5 -> 1, c: 16 -> 120

        self.full6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def print_x(self, x, s_r, e_r, s_c, e_c):
        for r in range(s_r, e_r + 1):
            for c in range(s_c, e_c + 1):
                print(x[r][c], end=" ")
            print("")

    def forward(self, x):
        # print(f"Input: {x.shape}")
        # print("conv1 weight:")
        # print(self.conv1.weight)
        # print("conv1 bias:")
        # print(self.conv1.bias)
        # print("x[0 -> 4][0 -> 4]: ")
        # self.print_x(x[0][0], 0, 2, 0, 2)
        x = self.conv1(x)
        # print("After conv1, x[0 -> 4][0 -> 4]: ")
        # self.print_x(x[0][0], 0, 2, 0, 2)
        # print(f"After conv1: {x.shape}")
        x = self.Sigmoid(x)
        x = self.pool2(x)
        # print(f"After pool2: {x.shape}")

        x = self.conv3(x)
        # print(f"After conv3: {x.shape}")
        x = self.Sigmoid(x)
        x = self.pool4(x)
        # print(f"After pool4: {x.shape}")

        x = self.conv5(x)
        # print(f"After conv5: {x.shape}")
        x = self.Flatten(x)
        # print(f"After Flatten: {x.shape}")

        x = self.full6(x)
        # print(f"After full6: {x.shape}")
        x = self.Sigmoid(x)
        x = self.output(x)
        # print(f"After output: {x.shape}")
        return x


if __name__ == "__main__":
    x = torch.rand([16, 1, 28, 28])
    print(f"x.shape = {x.shape}")
    model = LeNet()
    y = model(x)
