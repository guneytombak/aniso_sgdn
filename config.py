from src.utils import Container, DataName, ActivType, LossType
from datetime import datetime
import torch

'''
DIGITS_SIZE = 1797
ENERGY_SIZE = 768 | 8->h->2
GRID_SIZE = 10000 | 12->h->1
HOUSE_SIZE = 20640
IRIS_SIZE = 150
MNIST_SIZE = 10000
'''

cfg = Container()
cfg.online = True

# Run Parameters

cfg.seed = 42
cfg.dev = torch.device("cuda")
cfg.save_weights = False

# Dataset/Task Parameters

cfg.name = "default"
#cfg.date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
cfg.dataset_name = DataName.ENERGY

# Network Parameters

cfg.batch_size = 33
cfg.lr = 0.005

cfg.learn = (True, False)
cfg.rand_bound = 1
cfg.rand_step_size = 0.01

# if commented, gets the maximum: SIZE // batch_size
#cfg.maxiter = 38 

cfg.n_epochs = 30
cfg.per_epoch_test = 10

cfg.activ_type = (ActivType.RELU, ActivType.SIGMOID)
#cfg.loss_type = LossType.LNN

#cfg.input_size = 8
cfg.hidden_size = (128, [128, 16]) 
#cfg.output_size = 2

# Scheduler Parameters

cfg.sch__use = False

if cfg.sch__use:
    cfg.sch__gamma = 0.9
    cfg.sch__step_size = 20

