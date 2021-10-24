import random, os
import numpy as np
import torch
import copy

MNIST_SIZE = 1500
IRIS_SIZE = 150

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
    cfg.activation = getattr(cfg, 'activation', 'relu')
    
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
    
    if cfg.dataset_name.lower() == 'iris':
        cfg.input_size = 4
        cfg.output_size = 3
        cfg.hidden_size = getattr(cfg, 'hidden_size', 15)
        
        default_maxiter = IRIS_SIZE // cfg.batch_size
        cfg.maxiter = getattr(cfg, 'maxiter', default_maxiter)
        
    elif cfg.dataset_name.lower() == 'mnist':
        cfg.input_size = 28**2
        cfg.output_size = 10
        cfg.hidden_size = getattr(cfg, 'hidden_size', 256)
        
        default_maxiter = MNIST_SIZE // cfg.batch_size
        cfg.maxiter = getattr(cfg, 'maxiter', default_maxiter)

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