from src.utils import Container, DataName, ActivType, LossType
from datetime import datetime
import torch

'''
MNIST_SIZE = 10000
IRIS_SIZE = 150
ENERGY_SIZE = 768
'''

# Main Parameters

cfg = Container()

cfg.seed = 42

cfg.name = "mnist"
cfg.date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
cfg.dataset_name = DataName.MNIST

cfg.batch_size = 500
cfg.lr = 0.005
cfg.maxiter = 20

cfg.n_epochs = 100

cfg.activ_type = (ActivType.GELU, ActivType.RELU)
#cfg.loss_type = LossType.LNN

#cfg.input_size = 8
cfg.hidden_size = [100, 40]
#cfg.output_size = 2

# Scheduler Parameters

cfg.sch__use = False

if cfg.sch__use:
    cfg.sch__gamma = 0.9
    cfg.sch__step_size = 20

cfg.dev = torch.device("cuda")