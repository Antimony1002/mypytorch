import torch
from torch import nn

class Hydro(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

hydro = Hydro()
x = torch.tensor(1.0)
output = hydro(x)
print(output)