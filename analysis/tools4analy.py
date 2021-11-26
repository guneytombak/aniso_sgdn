import numpy as np
import matplotlib.pyplot as plt
import json
from regs import LinearRegression

import wandb

def lu_anly(name, del_init=0.1, 
            smooth_size=0.05, smooth_order=1,
            show_smoothing=True,
            save_folder='../results/figs/'):

    api = wandb.Api()
    run = api.run("guneytombak/aniso_sgdn/" + name)

    cfg = get_config(run)
    data = run.history()

    t = get_title(cfg)

    u = data['U'].to_numpy()
    l = data['D'].to_numpy()
    g = data['G'].to_numpy()

    loss = data['L'].to_numpy()
    #dh2 = data['Dh'].to_numpy()

    reg_l = LinearRegression(l, g, t+'lower', save_folder)
    if del_init:
        reg_l.del_initial(del_init)
    if smooth_size:
        reg_l.smooth(smooth_size, smooth_order, show=show_smoothing)
    reg_l.show()
    
    reg_u = LinearRegression(loss, g, t+'loss', save_folder)
    if del_init:
        reg_u.del_initial(del_init)
    if smooth_size:
        reg_u.smooth(smooth_size, smooth_order, show=show_smoothing)
    reg_u.show()

    return reg_l, reg_u

def get_config(run):

    dict_cfg = json.loads(run.json_config)
    cfg = dict()

    for key, value in dict_cfg.items():
        cfg[key] = value['value']

    return cfg


def get_title(cfg):

    t = ''

    if cfg['dataset_name'] == 'DataName.GRID':
        t += 'grid'
    elif cfg['dataset_name'] == 'DataName.ENERGY':
        t += 'energy'
    elif cfg['dataset_name'] == 'DataName.DIGITS':
        t += 'digits'    
    else:
        t += cfg['dataset_name']

    t += '_'

    if cfg['activ_type'] == 'ActivType.RELU':
        t += 'relu'
    elif cfg['activ_type'] == 'ActivType.SIGMOID':
        t += 'sigm'
    else:
        t += cfg['activ_type']

    t += '_'

    t += str(cfg['hidden_size'])

    t += '_'

    return t

