#%%

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from model import *
from config import cfg

#%%

# hyperparameters

torch.manual_seed(cfg.seed) # seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sch = cfg.sch

# model
model = Net(hidden_size=cfg.hidden_size).to(device)
optimizer = optim.SGD(model.parameters(), cfg.lr)

# data
transform=transforms.Compose([
    transforms.Resize((14,14)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

dataset = datasets.MNIST('../data', train=False, download=True,
                    transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, cfg.batch_size)

if sch.use:
    scheduler = StepLR(optimizer, step_size=sch.step_size, gamma=sch.gamma)

#%% Run the Code

model.initialize()
data_list = list()
for epoch in range(cfg.n_epochs):
    data = train(model, device, data_loader, 
                 optimizer, epoch, cfg.maxiter)
    data_list.append(data)
    if sch.use:
        scheduler.step()
    
# %%