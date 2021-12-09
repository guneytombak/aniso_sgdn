import sys

import torch
from torch import nn
from src.utils import ActivType, LossType

def select_loss(loss_type):
    
    if loss_type == LossType.NLL:
        return nn.CrossEntropyLoss()
    elif loss_type == LossType.MSE:
        mse = nn.MSELoss(reduction='mean')
        return lambda i, o : i.size(1)*mse(i, o)/2
    else:
        sys.exit(f'No loss function provided for \"{loss_type}\"')
    

def check_maxiter(maxiter, data_loader):
    
    possible_maxiter = len(data_loader.dataset) // data_loader.batch_size
    
    if maxiter > possible_maxiter:
        sys.exit(f'Change maxiter from {maxiter} to {possible_maxiter} or lower!')

    return maxiter

def random_update_function(p, p_grad, loss, cfg):
    
    dev = p.get_device()
    dev = dev if dev >= 0 else torch.device("cpu")
        
    dp = cfg.rand_step_size*torch.randn(p.shape).to(dev)

    p_new = p + dp

    p_new[p_new > cfg.rand_bound] = cfg.rand_bound
    p_new[p_new < -cfg.rand_bound] = -cfg.rand_bound
    
    return p_new

def trace(input, axis1=0, axis2=1):
    assert input.shape[axis1] == input.shape[axis2], input.shape

    shape = list(input.shape)
    strides = list(input.stride())
    strides[axis1] += strides[axis2]

    shape[axis2] = 1
    strides[axis2] = 0

    strided = torch.as_strided(input, size=shape, stride=strides)
    return strided.sum(dim=(axis1, axis2)).to(input.device)

