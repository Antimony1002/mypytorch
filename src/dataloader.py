import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10("./dataset", train=True, download=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter("DataLoader")

for epoch in range(2): 
    step=0    
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape, targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, global_step = step, dataformats="NCHW")
        step = step + 1

writer.close()