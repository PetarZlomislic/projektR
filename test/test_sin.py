import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

import numpy as np

class SinDataset(Dataset):

    def __init__(self):
        shape = (500,)
        l = np.random.uniform(-10, 10, shape) + 1.j * np.random.uniform(-10, 10, shape)
        l = [(x, np.sin(x)) for x in l]
        self.data = l

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, sin_x = self.data[idx]
        x  = torch.tensor([x])
        sin_x = torch.tensor([sin_x])
        return x, sin_x

batch_size = 64
train_loader = DataLoader(SinDataset(), batch_size=batch_size, shuffle=True)

class ComplexNet(nn.Module):
    
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.ln1 = ComplexLinear(1, 32)
        self.ln2 = ComplexLinear(32, 1)
             
    def forward(self,x):
        x = self.ln1(x)
        x = complex_relu(x)
        x = self.ln2(x)
        x = complex_relu(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ComplexNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def complex_mse_loss(output, target):
    return (0.5*(output - target)**2).mean(dtype=torch.complex64)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).type(torch.complex64), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        print(output, target)
        loss_o = loss(output, target)
        loss_o.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss_o.item())
            )

for epoch in range(1):
    train(model, device, train_loader, optimizer, epoch)