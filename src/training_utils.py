import sys

import torch
from torch import nn
from src.utils import ActivType, LossType

def id_func(arg1, *argv, **kwargs):
    """
    Dummy function: takes the first input and returns it.
    """
    return nn.Identity()(arg1)

def select_loss(loss_type):
    
    if loss_type == LossType.NLL:
        return nn.CrossEntropyLoss()
    elif loss_type == LossType.MSE:
        mse = nn.MSELoss(reduction='mean')
        # it should sum over all outputs, 
        # hence the mean should be multiplied by number of outputs 
        return lambda i, o : i.size(1)*mse(i, o)/2
    else:
        sys.exit(f'No loss function provided for \"{loss_type}\"')

def check_maxiter(maxiter, data_loader):
    
    possible_maxiter = len(data_loader.dataset) // data_loader.batch_size
    
    if maxiter > possible_maxiter:
        sys.exit(f'Change maxiter from {maxiter} to {possible_maxiter} or lower!')

    return maxiter

def random_update_function(p, rand_step_size, rand_bound=None):
    """
    Randomly updates by adding with a Gaussian random variable with 0 mean and rand_step_size
    The parameters are forced to be to be in the interval [-rand_bound, +rand_bound]
    """
    
    dev = p.get_device() # get the parameter device no
    dev = dev if dev >= 0 else torch.device("cpu") # if device no is less than zero, it is cpu
        
    dp = rand_step_size*torch.randn(p.shape).to(dev) # define random step vector
    p_new = p + dp # update the parameters with random step

    if rand_bound is not None:# force parameters to be in the interval [-rand_bound, +rand_bound]
        p_new[p_new > rand_bound] = rand_bound
        p_new[p_new < -rand_bound] = -rand_bound
    
    return p_new

def trace(input, axis1=0, axis2=1):
    """
    Trace method for specific axes.
    """
    assert input.shape[axis1] == input.shape[axis2], input.shape

    shape = list(input.shape)
    strides = list(input.stride())
    strides[axis1] += strides[axis2]

    shape[axis2] = 1
    strides[axis2] = 0

    strided = torch.as_strided(input, size=shape, stride=strides)
    return strided.sum(dim=(axis1, axis2)).to(input.device)

