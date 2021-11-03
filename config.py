from src.utils import Container, DataName, ActivType, LossType
from datetime import datetime
import torch

'''
DIGITS_SIZE = 1797
ENERGY_SIZE = 768
GRID_SIZE = 10000
HOUSE_SIZE = 20640
IRIS_SIZE = 150
MNIST_SIZE = 10000
'''

cfg = Container()

# Run Parameters

cfg.seed = (42, 78, 128, 733, 17)
cfg.dev = torch.device("cuda")
cfg.save_weights = False

# Dataset/Task Parameters

cfg.name = "default"
#cfg.date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
cfg.dataset_name = DataName.GRID

# Network Parameters

cfg.batch_size = 100
cfg.lr = 0.005

# if commented, gets the maximum: SIZE // batch_size
#cfg.maxiter = 38 

cfg.n_epochs = 100

cfg.activ_type = (ActivType.RELU, ActivType.SIGMOID)
#cfg.loss_type = LossType.LNN

#cfg.input_size = 8
cfg.hidden_size = (100, [100, 10])
#cfg.output_size = 2

# Scheduler Parameters

cfg.sch__use = False

if cfg.sch__use:
    cfg.sch__gamma = 0.9
    cfg.sch__step_size = 20

