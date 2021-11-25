#%%

import wandb
import pandas as pd
import numpy as np
import json

from tools import lu_anly
    

RUN_LIST = ['3aakt07w', '1fod6cpf', 
            '39yxdl9n', '2mgaycbu',
            '1fezwbit', '36kmvred',
            '20lpgr53', '3dp0p6gp']

for run_name in RUN_LIST:
    lu_anly(run_name)

# %%
