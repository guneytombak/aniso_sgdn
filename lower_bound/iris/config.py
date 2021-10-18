import numpy as np

class Container():
    pass

# Main Parameters

cfg = Container()

cfg.name = "Iris Dataset SGD"

cfg.batch_size = 5
cfg.lr = 0.005
cfg.maxiter = 35
cfg.seed = 42
cfg.n_epochs = 100
cfg.hidden_size = 50

cfg.input_size = 4
cfg.hidden_size = 15
cfg.output_size = 3

# Scheduler Parameters

sch = Container()

sch.use = True

if sch.use:
    sch.gamma = 0.9
    sch.step_size = 20

cfg.sch = sch

