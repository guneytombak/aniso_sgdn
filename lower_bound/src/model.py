import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from utils import ActivType, LossType, TaskType

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
            activation = cfg.activ_type
        
    return input_size, hidden_size, output_size, activation

def select_activation(activation):
    
    if activation == ActivType.GELU:
        return nn.GELU()
    elif activation ==  ActivType.RELU:
        return nn.ReLU()
    elif activation ==  ActivType.SIGMOID:
        return nn.Sigmoid()
    elif activation ==  ActivType.ID:
        return nn.Identity()
    else:
        print(f'No activation provided for \"{activation}\", using no activation.')
        return nn.Identity()

def select_loss(loss_type):
    
    if loss_type == LossType.NLL:
        return F.nll_loss
    elif loss_type == LossType.MSE:
        return F.mse_loss
    else:
        sys.exit(f'No loss function provided for \"{loss_type}\"')
    

def check_maxiter(maxiter, data_loader, epoch):
    
    possible_maxiter = len(data_loader.dataset) // data_loader.batch_size
    
    if maxiter > possible_maxiter:
        sys.exit(f'Change maxiter from {maxiter} to {possible_maxiter} or lower!')

    return maxiter

def select_out_func(cfg, task_type):

    if task_type is None:
        if cfg.task_type is not None:
            task_type = cfg.task_type
        else:
            sys.exit('No task type specified!')
        
    if task_type == TaskType.CLASSIFY:
        return F.log_softmax
    elif task_type == TaskType.REGRESS:

        id_in = nn.Identity()

        def id(arg1, *argv, **kwargs):
            return id_in(arg1)

        return id


class Net(nn.Module):
    def __init__(self, cfg=None, input_size=None, 
                 hidden_size=None, output_size=None,
                 activation=None, task_type=None, verbose=True):
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
        self.out_func = select_out_func(cfg, task_type)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc_in(x)
        x = self.act(x)
        
        for i in range(len(self.fc_hid)):
            x = self.fc_hid[i](x)
            x = self.act(x)
        
        x = self.fc_out(x)
        
        output = self.out_func(x, dim=1)
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


def eval_expectation(data_loader, model, device, optimizer, loss_func):
    
    grads = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # REVIEW
        output = model(data)
        loss = loss_func(output, target).mean()
        loss.backward()
        
        grads.append(model.get_grads())
    
    grads = torch.stack(grads, dim=0)
    delL = grads.mean(0)

    return delL
        
        
def train_epoch(model, cfg, data_loader, 
                optimizer, epoch, writer):
    
    loss_func = select_loss(cfg.loss_type)
    maxiter = check_maxiter(cfg.maxiter, data_loader, epoch)  
    
    model.train()
    
    delLs = list()
    losses = list()
    grads = list()
    
    gdl = list()
    
    eval_loader = copy.deepcopy(data_loader)
    
    for batch_idx, (data, target) in enumerate(data_loader):
        
        if batch_idx > (maxiter-1):
            break
        
        delL = eval_expectation(eval_loader, model, 
                                cfg.dev, optimizer, loss_func)
        delLs.append(delL)

        data, target = data.to(cfg.dev), target.to(cfg.dev)
        optimizer.zero_grad() # REVIEW
        output = model(data)
        loss = loss_func(output, target)
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
        
        gdl.append([gmDL_sq, g_sq, loss])
        
    epoch_loss = np.mean(np.array(losses))
        
    if epoch % 2 == 0:
        print(f'Epoch: {epoch} \tLoss: {epoch_loss:.6f}')
            
    grads = torch.stack(grads, dim=0)
    delLs = torch.stack(delLs, dim=0)
            
    output_data = dict()
    output_data['delL'] = np.array(delLs)
    output_data['grad'] = np.array(grads)
    output_data['loss'] = np.array(losses)
    output_data['GDL']  = np.array(gdl)
    
    return output_data, writer, epoch_loss