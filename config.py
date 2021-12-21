from src.utils import Container, DataName, ActivType, LossType
import torch

"""
DATASETS 
DIGITS  : 1797  |
ENERGY  : 768   | 8->h->2
GRID    : 10000 | 12->h->1
HOUSE   : 20640 |
IRIS    : 150   |
MNIST   : 10000 |
"""

cfg = Container() # struct-like class to contain run configurations  
cfg.online = True # W&B local or global

# Run Parameters

cfg.seed = 42
cfg.dev = torch.device("cuda")
cfg.save_weights = False

# Dataset/Task Parameters

cfg.experiment_name = ""
cfg.dataset_name = DataName.ENERGY

# Network Parameters

cfg.batch_size = 200
cfg.lr = 0.005

cfg.learn = (False, True)
cfg.rand_bound = 5
cfg.rand_step_size= 0.05

# if commented, gets the maximum: SIZE // batch_size
#cfg.maxiter = 38 

cfg.n_epochs = 200
cfg.per_epoch_test = 10

cfg.activ_type = (ActivType.RELU, ActivType.SIGMOID)
#cfg.loss_type = LossType.LNN

#cfg.input_size = automatically defined by src.utils.default_config 
cfg.hidden_size = (128, [128, 16], [128, 64, 16]) 
#cfg.output_size = automatically defined by src.utils.default_config 

# Scheduler Parameters

cfg.sch__use = False

if cfg.sch__use:
    cfg.sch__gamma = 0.9
    cfg.sch__step_size = 20

