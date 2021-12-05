#%%

import wandb
import pandas as pd
import numpy as np
import json

from tools4analy import lu_anly


LEARN_LIST = []
save_folder_learn = "../results/figs/learn/"
RAND_LIST = []
save_folder_rand = "../results/figs/rand/"

#%%

for run_name in LEARN_LIST:
    lu_anly(run_name, save_folder=save_folder_learn)

# %%

for run_name in RAND_LIST:
    lu_anly(run_name, save_folder=save_folder_rand)
