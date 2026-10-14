# 本文件展示了卷积操作
import torch
import torch.nn.functional as F

input = torch.tensor([[1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,0,0],
                     [5,2,3,1,1],
                     [2,1,0,1,1]])
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

# 改换shape以适应卷积操作，1 batch size，1 channel
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)

# stride是kernel每次平移的步长
output = F.conv2d(input, kernel, stride=1)
print(output)

output2 = F.conv2d(input, kernel, stride=2)
print(output2)

# padding，提高边缘数据的利用率，防止越卷越小的情况
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)
