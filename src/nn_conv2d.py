import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

# 这里是导入数据集
dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, 
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size = 64)
step = 0
writer = SummaryWriter("./logs")

# 这里是神经网络部分，继承自nn.Module
class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        # 卷积操作
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

my = myModel()
for data in dataloader:
    imgs, targets = data
    output = my(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)

    output = torch.reshape(output, (-1, 3, output.size(2), output.size(3)))
    writer.add_images("output", output, step)
    step += 1