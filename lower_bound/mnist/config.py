import numpy as np

class Container():
    pass

# Main Parameters

cfg = Container()

cfg.name = "MNIST Dataset SGD"

cfg.batch_size = 20
cfg.lr = 0.005
cfg.maxiter = 500
cfg.seed = 42
cfg.n_epochs = 5
cfg.hidden_size = 50

# Scheduler Parameters

sch = Container()

sch.use = True

if sch.use:
    sch.gamma = 0.9
    sch.step_size = 20

cfg.sch = sch

