import torch
import torch.nn.functional as F
from torch import nn

class Centeredlayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = Centeredlayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))


net = nn.Sequential(nn.Linear(8, 128), Centeredlayer())

Y = net(torch.rand(4, 8))
print(Y.mean())



















