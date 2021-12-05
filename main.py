import pickle5 as pickle
from datetime import datetime

from config import cfg
from src.run import run
from src.utils import cfg_definer

import wandb

cfgs_fin = list()

def main(cfg):
    
    cfg_list = cfg_definer(cfg)
    n_cfgs = len(cfg_list)
    for ind, cfg in enumerate(cfg_list):
        print(f"RUNNING CONFIG NO {ind+1}/{n_cfgs}")

        if cfg.online:
            wb_run = wandb.init(reinit=True, config=cfg.to_dict())
        else:
            wb_run = wandb.init(reinit=True, config=cfg.to_dict(), mode="disabled")
        
        run(cfg)
        #cfg.finish_date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        wandb.config.update(cfg.to_dict())
        wb_run.finish()

    
if __name__ == '__main__':
    
    main(cfg)