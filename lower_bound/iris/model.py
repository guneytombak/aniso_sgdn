import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def default_sizes(input_size, hidden_size, output_size, cfg):
    
    if cfg is not None:
    
        if input_size is None:
            input_size = cfg.input_size
        
        if hidden_size is None:
            hidden_size = cfg.hidden_size
            
        if output_size is None:
            output_size = cfg.output_size   
               
    else:
        
        if input_size is None:
            input_size = 4
        
        if hidden_size is None:
            hidden_size = 50
            
        if output_size is None:
            output_size = 3
        
    return input_size, hidden_size, output_size


def check_maxiter(maxiter, data_loader, epoch):
    
    possible_maxiter = len(data_loader.dataset) // data_loader.batch_size
    
    if maxiter > possible_maxiter:
        if epoch == 0:
            print(f'Maxiter is changed from {maxiter} to {possible_maxiter}')
        maxiter = possible_maxiter

    return maxiter

class Net(nn.Module):
    def __init__(self, cfg=None, input_size=None, 
                 hidden_size=None, output_size=None,
                 verbose=True):
        super(Net, self).__init__()
        
        ins, his, ous = default_sizes(input_size, hidden_size, 
                                      output_size, cfg)
        if verbose:
            print(f'Model dims are {ins}->{his}->{ous}')
        
        self.fc1 = nn.Linear(ins, his)
        self.fc2 = nn.Linear(his, ous)
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

def eval_expectation(data_loader, model, device, optimizer):
    
    s = 0
    grads = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # REVIEW
        output = model(data)
        loss = F.nll_loss(output, target).mean()
        loss.backward()
        
        grads.append(model.get_grads())
    
    grads = torch.stack(grads, dim=0)
    delL = grads.mean(0)

    return delL
        

def train(model, device, data_loader, 
          optimizer, epoch, maxiter, writer):
    
    maxiter = check_maxiter(maxiter, data_loader, epoch)  
    
    model.train()
    
    delLs = list()
    losses = list()
    grads = list()
    
    for batch_idx, (data, target) in enumerate(data_loader):
        
        if batch_idx > (maxiter-1):
            break
        
        eval_loader = copy.deepcopy(data_loader)
        delL = eval_expectation(eval_loader, model, 
                                device, optimizer)
        delLs.append(delL)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # REVIEW
        output = model(data)
        loss = F.nll_loss(output, target)
        losses.append(np.float64(loss))
        loss.backward()
        grad = model.get_grads()
        grads.append(grad)
        optimizer.step()
        
        g_sq = np.float64(torch.sum(torch.square(grad)))
        gmDL_sq = np.float64(torch.sum(torch.square(grad - delL)))
        loss = np.float64(loss)
        iter_no = batch_idx + epoch*maxiter
        
        writer.add_scalars('main', {'(g-DL)^2'  : gmDL_sq,
                                    'g^2'       : g_sq,
                                    'L'         : loss}, iter_no)
        
    epoch_loss = np.mean(np.array(losses))
        
    if epoch % 10 == 0:
        print(f'Epoch: {epoch} \tLoss: {epoch_loss:.6f}')
            
    grads = torch.stack(grads, dim=0)
    delLs = torch.stack(delLs, dim=0)
            
    data = dict()
    data['delL'] = np.array(delLs)
    data['grad'] = np.array(grads)
    data['loss'] = np.array(losses)
    
    return data, writer