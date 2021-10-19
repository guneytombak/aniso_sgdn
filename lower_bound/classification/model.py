import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def default_model(input_size, hidden_size, output_size, 
                  activation, cfg):
    
    if cfg is not None:
    
        if input_size is None:
            input_size = cfg.input_size
        
        if hidden_size is None:
            hidden_size = cfg.hidden_size
            
        if output_size is None:
            output_size = cfg.output_size
            
        if activation is None:
            activation = cfg.activation
        
    return input_size, hidden_size, output_size, activation

def select_activation(activation):
    
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation.lower() == 'id':
        return nn.Identity()
    else:
        print(f'No activation provided for \"{activation}\", using no activation.')
        return nn.Identity()

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
                 activation=None, verbose=True):
        super(Net, self).__init__()
        
        ins, his, ous, acv = default_model(input_size, hidden_size, output_size, 
                                           activation, cfg)
        if verbose:
            print(f'Model dims are {ins}->{his}->{ous}')
        
        if not isinstance(his, list):
            his = [his]
        
        self.fc_hid = list()
        
        self.fc_in = nn.Linear(ins, his[0])
        
        for i in range(len(his)-1):
            self.fc_hid.append(nn.Linear(his[i], his[i+1]))
        
        self.fc_hid = nn.ModuleList(self.fc_hid)
        
        self.fc_out = nn.Linear(his[-1], ous)
        
        self.act = select_activation(acv)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc_in(x)
        x = self.act(x)
        
        for i in range(len(self.fc_hid)):
            x = self.fc_hid[i](x)
            x = self.act(x)
        
        x = self.fc_out(x)
        
        output = F.log_softmax(x, dim=1)
        return output
    
    def initialize(self):
        nn.init.xavier_uniform_(self.fc_in.weight)
        for i in range(len(self.fc_hid)):
            nn.init.xavier_uniform_(self.fc_hid[i].weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        
    def get_weights(self):
        
        w = self.fc_in.weight.cpu().view(-1).clone()
        for i in range(len(self.fc_hid)):
            w_i = self.fc_hid[i].weight.cpu().view(-1).clone()
            w = torch.cat((w, w_i), 0)
        w_out = self.fc_out.weight.cpu().view(-1).clone()
        return torch.cat((w, w_out), 0)
    
    def get_grads(self):
        g = self.fc_in.weight._grad.cpu().view(-1).clone()
        for i in range(len(self.fc_hid)):
            g_i = self.fc_hid[i].weight._grad.cpu().view(-1).clone()
            g = torch.cat((g, g_i), 0)
        g_out = self.fc_out.weight._grad.cpu().view(-1).clone()
        return torch.cat((g, g_out), 0)


def eval_expectation(data_loader, model, device, optimizer):
    
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
        
        
def train_epoch(model, device, data_loader, 
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
    
    return data, writer, epoch_loss