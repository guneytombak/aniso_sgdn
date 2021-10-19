#%%

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from data import get_data

from torch.utils.tensorboard import SummaryWriter

from utils import seed_everything, default_config
from model import *
from config import cfg

#%% Parameters

# hyperparameters

cfg = default_config(cfg)
sch = cfg.sch
seed_everything(cfg.seed)

writer = SummaryWriter(cfg.name + cfg.date + '.sw')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = Net(cfg=cfg).to(device)
optimizer = optim.SGD(model.parameters(), cfg.lr)

if sch.use:
    scheduler = StepLR(optimizer, step_size=sch.step_size, gamma=sch.gamma)

# data

dataset = get_data(cfg.dataset_name)
data_loader = torch.utils.data.DataLoader(dataset, 
                                          cfg.batch_size,
                                          shuffle=True)

#%% Run the Model

model.initialize()
model.to(device)
data_list = list()
for epoch in range(cfg.n_epochs):
    data, writer, epoch_loss = train_epoch(model, device, data_loader, 
                               optimizer, epoch, cfg.maxiter, writer)
    data_list.append(data)
    if sch.use:
        scheduler.step()
    
writer.add_hparams(cfg.to_dict(), {'loss': epoch_loss})
    
writer.close()
