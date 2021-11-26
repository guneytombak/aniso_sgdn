#%%

import wandb
import pandas as pd
import numpy as np
import json

from tools4analy import lu_anly
    

LEARN_LIST = ['3aakt07w', '1fod6cpf', 
            '39yxdl9n', '2mgaycbu',
            '1fezwbit', '36kmvred',
            '20lpgr53', '3dp0p6gp']

RAND_LIST = ['30hjr48r', '1ytnagmh',
             '2eg2ol1u', '3n582whz',
             '3sweuw8v', 'hm1j1njg',
             'y8b7c0ji', 'kgozjquh']

#%%

for run_name in LEARN_LIST:
    lu_anly(run_name, save_folder="../results/figs/learn/")

# %%

for run_name in RAND_LIST:
    lu_anly(run_name, save_folder="../results/figs/rand/")
# %%
