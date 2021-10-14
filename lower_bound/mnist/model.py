import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

def default_sizes(input_size, hidden_size, output_size):
    
    if input_size is None:
        input_size = 14**2
    
    if hidden_size is None:
        hidden_size = 1024
        
    if output_size is None:
        output_size = 10
        
    return input_size, hidden_size, output_size

class Net(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, output_size=None):
        super(Net, self).__init__()
        input_size, hidden_size, output_size = default_sizes(input_size, hidden_size, output_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def initialize(self):
        self.fc1.weight = nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2.weight = nn.init.xavier_uniform_(self.fc2.weight)
        
    def get_weights(self):
        w1 = self.fc1.weight.cpu().view(-1).clone()
        w2 = self.fc2.weight.cpu().view(-1).clone()
        return torch.cat((w1, w2), 0)
    
    def get_grads(self):
        g1 = self.fc1.weight._grad.cpu().view(-1).clone()
        g2 = self.fc2.weight._grad.cpu().view(-1).clone()
        return torch.cat((g1, g2), 0)

def eval_expectation(data_loader, model, device):
    grads = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target).mean()
        loss.backward()
        
        grads.append(model.get_grads())
    
    grads = torch.stack(grads, dim=0)
    
    return grads
        

def train(model, device, data_loader, optimizer, epoch, maxiter):
    model.train()
    weights_list = list() 
    grads_list = list()
    batch_no = 0
    idxs = np.random.choice(len(data_loader), 
                     maxiter, replace=False)
    
    for batch_idx, (data, target) in enumerate(data_loader):
        
        if batch_idx not in idxs:
            continue
        
        grads = eval_expectation(data_loader, model, device)
        weights_list.append(model.get_weights())
        grads_list.append(grads)
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        batch_no += 1
        
        if batch_no % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_no}/{maxiter}]\tLoss: {loss:.6f}')
            
    data = dict()
    data['weights'] = np.stack(weights_list)
    data['grads'] = np.stack(grads_list)
    
    return data