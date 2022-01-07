import random, os
import numpy as np
import torch
import copy

# Number of samples of datasets

DIGITS_SIZE = 1797
ENERGY_SIZE = 768
GRID_SIZE = 10000
HOUSE_SIZE = 19794
IRIS_SIZE = 150
MNIST_SIZE = 10000

from enum import Enum, auto
import math

class TaskType(Enum):
    """
    Task types used
    """
    REGRESS = auto()
    CLASSIFY = auto()


class ActivType(Enum):
    """
    Activation functions used
    """
    ID = auto()
    SIGMOID = auto()
    RELU = auto()
    GELU = auto()
    

class LossType(Enum):
    """
    Loss functions used
    NLL for classification
    MSE for regression
    """
    NLL = auto()
    MSE = auto()


class DataName(Enum):
    """
    Dataset names
    """
    IRIS = auto()
    MNIST = auto()
    ENERGY = auto()
    GRID = auto()
    HOUSE = auto()
    DIGITS = auto()


class Container():
    """
    A container class similar to Matlab's structs.
    It can be turned into a dictionary using to_dict method.
    """
    def __init__(self):
        self.name = None
        
    def __print__(self):
        print(self.to_dict())
        if self.name is None:
            return "Untitled Container"
        else:
            return f"{self.name} Container"
    
    def to_dict(self, tb_comp=False):
        class_vars = vars(Container)
        inst_vars = vars(self)
        all_vars = dict(class_vars)
        all_vars.update(inst_vars)
        public_vars = {k: v for k, v in all_vars.items() if not k.startswith('_')}
        del public_vars['to_dict']
        
        sub_data, sub_names = list(), list()
        
        for key, value in public_vars.items():
            if isinstance(value, Container):
                subcontainer = value.to_dict()
                sub_data.append({f'{key}__{k}': v for k, v in subcontainer.items()})
                sub_names.append(key)
                
        for sub_datum, sub_name in zip(sub_data, sub_names):
            public_vars.update(sub_datum)
            del public_vars[sub_name]
                
        if tb_comp:
            for key, value in public_vars.items():
                if isinstance(value, list):
                    public_vars[key] = torch.Tensor(value)
                
        return public_vars


def seed_everything(seed: int):
    """
    Seed method for PyTorch for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def default_config(cfg):
    """
    This function changes the configuration parameters 
    according to the corresponding dataset needs:
    predefined by dataset: input_size, output_size, task_type, loss_type, maxiter
    randomly defined: batch_size, lr, n_epochs, hidden_size, sch parameters    
    """
    
    cfg.seed = getattr(cfg, 'seed', 42)
    
    cfg.batch_size = getattr(cfg, 'batch_size', 4)
    cfg.lr = getattr(cfg, 'lr', 0.005)
    cfg.n_epochs = getattr(cfg, 'n_epochs', 100)
    
    if not hasattr(cfg, 'sch'):
        sch = Container()
        sch.use = False
        cfg.sch = sch
    
    if cfg.sch.use == True:
        cfg.sch.gamma = getattr(cfg.sch, 'gamma', 0.9)
        default_step_size = cfg.n_epochs // 10
        cfg.sch.step_size = getattr(cfg.sch, 'step_size', default_step_size)
    else:
        cfg.sch.gamma = None
        cfg.sch.step_size = None
    
    if cfg.dataset_name == DataName.DIGITS:

        cfg.task_type = TaskType.CLASSIFY

        cfg.input_size = 64
        cfg.output_size = 10
        cfg.hidden_size = getattr(cfg, 'hidden_size', 50)
        
        default_maxiter = DIGITS_SIZE // cfg.batch_size
        cfg.maxiter = getattr(cfg, 'maxiter', default_maxiter)
        
    elif cfg.dataset_name == DataName.ENERGY:

        cfg.task_type = TaskType.REGRESS

        cfg.input_size = 8
        cfg.output_size = 2
        cfg.hidden_size = getattr(cfg, 'hidden_size', 50)
        
        default_maxiter = ENERGY_SIZE // cfg.batch_size
        cfg.maxiter = getattr(cfg, 'maxiter', default_maxiter)
    
    elif cfg.dataset_name == DataName.GRID:

        cfg.task_type = TaskType.REGRESS

        cfg.input_size = 12
        cfg.output_size = 1
        cfg.hidden_size = getattr(cfg, 'hidden_size', 50)
        
        default_maxiter = GRID_SIZE // cfg.batch_size
        cfg.maxiter = getattr(cfg, 'maxiter', default_maxiter)

    elif cfg.dataset_name == DataName.HOUSE:

        cfg.task_type = TaskType.REGRESS

        cfg.input_size = 8
        cfg.output_size = 1
        cfg.hidden_size = getattr(cfg, 'hidden_size', 50)
        
        default_maxiter = HOUSE_SIZE // cfg.batch_size
        cfg.maxiter = getattr(cfg, 'maxiter', default_maxiter)
        
    elif cfg.dataset_name == DataName.IRIS:

        cfg.task_type = TaskType.CLASSIFY

        cfg.input_size = 4
        cfg.output_size = 3
        cfg.hidden_size = getattr(cfg, 'hidden_size', 15)

        
        default_maxiter = IRIS_SIZE // cfg.batch_size
        cfg.maxiter = getattr(cfg, 'maxiter', default_maxiter)
        
    elif cfg.dataset_name == DataName.MNIST:

        cfg.task_type = TaskType.CLASSIFY

        cfg.input_size = 28**2
        cfg.output_size = 10
        cfg.hidden_size = getattr(cfg, 'hidden_size', 256)
        
        default_maxiter = MNIST_SIZE // cfg.batch_size
        cfg.maxiter = getattr(cfg, 'maxiter', default_maxiter)

    # Generated Dataset
    elif isinstance(cfg.dataset_name, dict):

        if 'task_type' in cfg.dataset_name:
            cfg.task_type = cfg.dataset_name['task_type']
        else:
            cfg.task_type = TaskType.REGRESS

        cfg.input_size = cfg.dataset_name['input_size']
        cfg.output_size = cfg.dataset_name['output_size']

        default_maxiter = cfg.dataset_name['sample_size'] // cfg.batch_size
        cfg.maxiter = getattr(cfg, 'maxiter', default_maxiter)

    if cfg.task_type == TaskType.CLASSIFY:
        cfg.loss_type = LossType.NLL
    elif cfg.task_type == TaskType.REGRESS:
        cfg.loss_type = LossType.MSE
        
    print(f"Maxiter is {cfg.maxiter}/{default_maxiter}")

    cfg = cfg_renamer(cfg) # change the experiment name according to the configuration parameters
    
    return cfg

def cfg_renamer(cfg):
    """
    If the experiment_name is '', cfg_renamer changes it to format:
    <dataset_name>[<hidden_sizes>]<activation_type>_<learning:l/r>
    e.g.: grid[128, 16]relu_L
    """

    if isinstance(cfg.dataset_name, DataName): 
        exp_name = str(cfg.dataset_name).split('.')[1].lower()
    elif isinstance(cfg.dataset_name, dict):
        exp_name = 'gen' + str(cfg.dataset_name['sample_size'])

    exp_name += str(cfg.hidden_size) if isinstance(cfg.hidden_size, list) else '[' + str(cfg.hidden_size) + ']'
    exp_name += str(cfg.activ_type).split('.')[1].lower()
    exp_name += '_l' if cfg.learn else '_r'

    cfg.experiment_name += '_' + exp_name

    return cfg

def cfg_definer(cfgs):
    """
    Creates a branch for each tuple of parameters which constructs a tree of configurations.
    So, it constructs all possible configurations for parameters as p = (a,b,...)
    """
    
    cfg_list = [copy.deepcopy(cfgs)]
    
    attrs = dict()

    for k, v in cfgs.to_dict().items():
        if isinstance(v, tuple):
            attrs[k] = v
            
    for key, possible_values in attrs.items():
        cfg_list_new = list()
        for cfg in cfg_list:
            for value in possible_values:
                cfg_new = copy.deepcopy(cfg)
                setattr(cfg_new, key, value)
                cfg_list_new.append(cfg_new)

        cfg_list = cfg_list_new
        
    return cfg_list

