#%%

from utils import Container
from datetime import datetime

# Main Parameters

cfg = Container()

cfg.name = "iris"
cfg.date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
cfg.dataset_name = 'iris'

cfg.batch_size = 5
cfg.lr = 0.005
cfg.maxiter = 30
cfg.seed = 42
cfg.n_epochs = 100

cfg.activation = ('gelu', 'relu')

cfg.input_size = 4
cfg.hidden_size = (10, 20, [10, 20])
cfg.output_size = 3

# Scheduler Parameters

sch = Container()

sch.use = False

if sch.use:
    sch.gamma = 0.9
    sch.step_size = 20

cfg.sch = sch


# %%
