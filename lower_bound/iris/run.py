#%%

import torch
import torch.optim as optim
from customDataset import CustomDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

from model import *
from config import cfg

#%%

writer = SummaryWriter('iris.sw')

# hyperparameters

torch.manual_seed(cfg.seed) # seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sch = cfg.sch

# model
model = Net(cfg=cfg).to(device)
optimizer = optim.SGD(model.parameters(), cfg.lr)

if sch.use:
    scheduler = StepLR(optimizer, step_size=sch.step_size, gamma=sch.gamma)

# data

iris = load_iris()
scal = StandardScaler()
dataset = CustomDataset(scal.fit_transform(iris.data), iris.target)

data_loader = torch.utils.data.DataLoader(dataset, 
                                          cfg.batch_size,
                                          shuffle=True)

#%% Run the Code

model.initialize()
data_list = list()
for epoch in range(cfg.n_epochs):
    data, writer = train(model, device, data_loader, 
                         optimizer, epoch, cfg.maxiter, writer)
    data_list.append(data)
    if sch.use:
        scheduler.step()
    
writer.close()
# %%

s = 0

for batch_idx, (data, target) in enumerate(data_loader):
    
    print(target)
    s += np.sum(np.array(target))
    
print(s)

