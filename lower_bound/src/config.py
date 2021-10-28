from utils import Container, DataName, ActivType, LossType
from datetime import datetime

'''
MNIST_SIZE = 1500
IRIS_SIZE = 150
ENERGY_SIZE = 768
'''

# Main Parameters

cfg = Container()

cfg.name = "iris"
cfg.date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
cfg.dataset_name = DataName.IRIS

cfg.batch_size = 5
cfg.lr = 0.005
cfg.maxiter = 30
cfg.seed = 42
cfg.n_epochs = 100

cfg.activ_type = (ActivType.GELU, ActivType.RELU)
#cfg.loss_type = LossType.LNN

#cfg.input_size = 8
cfg.hidden_size = 40
#cfg.output_size = 2

# Scheduler Parameters

cfg.sch__use = False

if cfg.sch__use:
    cfg.sch__gamma = 0.9
    cfg.sch__step_size = 20

