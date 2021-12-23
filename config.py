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
# if cfg.experiment_name="", automatically defined as 
# <dataset_name>[<hidden_sizes>]<activation_type>_<learning:l/r>
cfg.dataset_name = DataName.ENERGY

# Network Parameters

cfg.batch_size = 33
cfg.lr = 0.005

cfg.learn = (False, True)
cfg.rand_bound = 1
cfg.rand_step_size= 0.1

#cfg.maxiter = if commented, gets the maximum: SIZE // batch_size

cfg.n_epochs = 100
cfg.per_epoch_test = 10

cfg.activ_type = (ActivType.RELU, ActivType.SIGMOID)
#cfg.loss_type = automatically defined by dataset_name / task_type

#cfg.input_size = automatically defined by src.utils.default_config 
cfg.hidden_size = (128, [128, 16], [128, 64, 16]) 
#cfg.output_size = automatically defined by src.utils.default_config 

# Scheduler Parameters for Learning via SGD

cfg.sch__use = False

if cfg.sch__use:
    cfg.sch__gamma = 0.9
    cfg.sch__step_size = 20

