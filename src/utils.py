import random, os
import numpy as np
import torch
import copy

DIGITS_SIZE = 1797
ENERGY_SIZE = 768
GRID_SIZE = 10000
HOUSE_SIZE = 20640
IRIS_SIZE = 150
MNIST_SIZE = 10000

from enum import Enum, auto

class TaskType(Enum):
    REGRESS = auto()
    CLASSIFY = auto()


class ActivType(Enum):
    ID = auto()
    SIGMOID = auto()
    RELU = auto()
    GELU = auto()
    

class LossType(Enum):
    NLL = auto()
    MSE = auto()


class DataName(Enum):
    IRIS = auto()
    MNIST = auto()
    ENERGY = auto()
    GRID = auto()
    HOUSE = auto()
    DIGITS = auto()


class Container():
    
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
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def default_config(cfg):
    
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
        cfg.loss_type = LossType.NLL
        
        default_maxiter = DIGITS_SIZE // cfg.batch_size
        cfg.maxiter = getattr(cfg, 'maxiter', default_maxiter)
        
    elif cfg.dataset_name == DataName.ENERGY:

        cfg.task_type = TaskType.REGRESS

        cfg.input_size = 8
        cfg.output_size = 2
        cfg.hidden_size = getattr(cfg, 'hidden_size', 50)
        cfg.loss_type = LossType.MSE
        
        default_maxiter = ENERGY_SIZE // cfg.batch_size
        cfg.maxiter = getattr(cfg, 'maxiter', default_maxiter)
    
    elif cfg.dataset_name == DataName.GRID:

        cfg.task_type = TaskType.REGRESS

        cfg.input_size = 12
        cfg.output_size = 1
        cfg.hidden_size = getattr(cfg, 'hidden_size', 50)
        cfg.loss_type = LossType.MSE
        
        default_maxiter = GRID_SIZE // cfg.batch_size
        cfg.maxiter = getattr(cfg, 'maxiter', default_maxiter)

    elif cfg.dataset_name == DataName.HOUSE:

        cfg.task_type = TaskType.REGRESS

        cfg.input_size = 8
        cfg.output_size = 1
        cfg.hidden_size = getattr(cfg, 'hidden_size', 50)
        cfg.loss_type = LossType.MSE
        
        default_maxiter = HOUSE_SIZE // cfg.batch_size
        cfg.maxiter = getattr(cfg, 'maxiter', default_maxiter)
        
    elif cfg.dataset_name == DataName.IRIS:

        cfg.task_type = TaskType.CLASSIFY

        cfg.input_size = 4
        cfg.output_size = 3
        cfg.hidden_size = getattr(cfg, 'hidden_size', 15)
        cfg.loss_type = LossType.NLL
        
        default_maxiter = IRIS_SIZE // cfg.batch_size
        cfg.maxiter = getattr(cfg, 'maxiter', default_maxiter)
        
    elif cfg.dataset_name == DataName.MNIST:

        cfg.task_type = TaskType.CLASSIFY

        cfg.input_size = 28**2
        cfg.output_size = 10
        cfg.hidden_size = getattr(cfg, 'hidden_size', 256)
        cfg.loss_type = LossType.NLL
        
        default_maxiter = MNIST_SIZE // cfg.batch_size
        cfg.maxiter = getattr(cfg, 'maxiter', default_maxiter)

    print(f"Maxiter is {cfg.maxiter}/{default_maxiter}")
    
    return cfg

def ld2dl(ld):
    return {k: [dic[k] for dic in ld] for k in ld[0]}

def cfg_definer(cfgs):
    
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