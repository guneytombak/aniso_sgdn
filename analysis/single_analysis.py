# %%
###############################################

import pandas as pd 
import wandb
import numpy as np
from matplotlib import pyplot as plt

api = wandb.Api()

#%% 

def get_run_plot(id, x_name, y_name, log=True,
                 prefix="guneytombak/aniso_sgdn/",
                 ax=None, label=None):

    label = label if label is not None else y_name

    run = api.run(prefix + id) 

    data = np.array(run.history()[[x_name, y_name]])
    data = data[~np.isnan(data).any(axis=1)]
    x, y = data[:,0], data[:,1]
    ind_sort = np.argsort(x, axis=0)
    x, y = x[ind_sort], y[ind_sort]

    if log:
        if ax is None:
            plt.xscale("log", base=10)
            plt.yscale("log", base=10) 
            plt.loglog(x,y,label=label)
        else:
            ax.set_xscale("log", base=10) 
            ax.set_yscale("log", base=10)
            ax.loglog(x,y,label=label)
            return ax
    else:
        if ax is None:
            plt.plot(x,y,label=label)
        else:
            ax.plot(x,y,label=label)
            return ax

#%%

ids2check = ['2ws40omy', 'j80pri8k', '1a3oohh8', '2lihcqa9']

def blower(s):
    s = s.lower()
    return s[0].upper() + s[1:]

title_dict = {'gen1'    : 'Generated Dataset \#1',
              'gen2'    : 'Generated Dataset \#2',
              'grid'    : 'Simulated Electrical Grid Stability Dataset',
              'energy'  : 'Energy Efficiency Dataset'}

def get_dname(cfg):
    if isinstance(cfg['dataset_name'], dict):
        if cfg['dataset_name']['sample_size'] == 2000:
            return 'gen2'
        else:
            return 'gen1'
    else: 
        return str(cfg['dataset_name']).split('.')[1].lower()

def titler(run):
    title = title_dict[get_dname(run.config)]
    title += ' ' + blower(str(run.config['activ_type']).split('.')[1])

    if isinstance(run.config['hidden_size'], list):
        title += ' ' + str(run.config['hidden_size'])
    else:
        title += ' [' + str(run.config['hidden_size']) + ']'
    title += ' SGD' if run.name[-10] == 'l' else ' RW'

    return title


def plot_lossVS(id, log=True, save=True):

    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    fig.set_dpi(200)
    get_run_plot(id, 'Loss', 'EoSq', log=log, ax=ax, label='Expectation of Square of Gradient')
    get_run_plot(id, 'Loss', 'Lower', log=log, ax=ax, label='Lower Bound')
    ax.set_xlabel('Loss', size=14)

    prefix="guneytombak/aniso_sgdn/"
    
    ax.set_title(titler(api.run(prefix + id)))
    #ax.legend()
    if save:
        fig.savefig(f'results/id{id}.pdf')

    
# %%
for id in ids2check:
    plot_lossVS(id, True)

# %%
