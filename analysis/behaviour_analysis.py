#%% 

import pandas as pd 
import wandb
import numpy as np
from matplotlib import pyplot as plt

api = wandb.Api()
entity, project = "guneytombak", "aniso_sgdn"  # set to your entity and project 
runs = api.runs(entity + "/" + project) 

#%%

def get_size(cfg):
    if isinstance(cfg['hidden_size'], int):
        return 1
    else:
        return len(cfg['hidden_size'])

def get_dname(cfg):
    if isinstance(cfg['dataset_name'], dict):
        if cfg['dataset_name']['sample_size'] == 2000:
            return 'gen2'
        else:
            return 'gen1'
    else: 
        return str(cfg['dataset_name']).split('.')[1].lower()

def get_config(run):
    c = dict()
    # hidden size
    c['hsize'] = get_size(run.config)
    # dataset name
    c['dname'] = get_dname(run.config)
    # activation type
    c['activ'] = str(run.config['activ_type']).split('.')[1].lower()
    # learning / random
    c['learn'] = True if run.name[-10] == 'l' else False

    return c

def blower(s):
    s = s.lower()
    return s[0].upper() + s[1:]

def run_labeler(name):
    n = '[' + name.split('[')[1].split('_')[0]
    return blower(n.split(']')[1]) + ' ' + n.split(']')[0] + ']'

title_dict = {'gen1'    : 'Generated Dataset #1',
              'gen2'    : 'Generated Dataset #2',
              'grid'    : 'Simulated Electrical Grid Stability Dataset',
              'energy'  : 'Energy Efficiency Dataset'}
    
def titler(run):
    title = title_dict[get_dname(run.config)]
    title += ' SGD' if run.name[-10] == 'l' else ' RW'

    return title

def get_run_plot(run, x_name='Loss', y_name='EoSq', log=True,
                 ax=None, label=None, alpha=0.7):

    label = label if label is not None else run_labeler(run.name)

    data = np.array(run.history()[[x_name, y_name]])
    data = data[~np.isnan(data).any(axis=1)]
    x, y = data[:,0], data[:,1]
    ind_sort = np.argsort(x, axis=0)
    x, y = x[ind_sort], y[ind_sort]

    if log:
        if ax is None:
            plt.xscale("log", base=10)
            plt.yscale("log", base=10) 
            plt.loglog(x,y,label=label,alpha=alpha)
        else:
            ax.set_xscale("log", base=10) 
            ax.set_yscale("log", base=10)
            ax.loglog(x,y,label=label,alpha=alpha)
            return ax
    else:
        if ax is None:
            plt.plot(x,y,label=label,alpha=alpha)
        else:
            ax.plot(x,y,label=label,alpha=alpha)
            return ax

#%%
set_of_runs = dict()

for run in runs: 
    if 'final' in run.TAGS and run.state == 'finished' and run.config['seed'] == 42:
        data_dict = get_config(run)
        lr = 'l' if data_dict['learn'] else 'r'
        set_name = data_dict['dname'] + lr

        if set_name not in set_of_runs:
            set_of_runs[set_name] = list()

        set_of_runs[set_name].append(run)
            
#%% 

for sor in set_of_runs:
    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=False)
    fig.set_dpi(150)
    for run in set_of_runs[sor]:
        get_run_plot(run, ax=ax)

    ax.set_title(titler(run), size=14)
    ax.grid()
    ax.set_xlabel('Loss', size=14)
    ax.set_ylabel('Expectation of the Square of Gradient', size=14)
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    fig.savefig(f'results/bhv_{sor}.pdf')


# %%
