from config import cfg
from src.run import run
from src.utils import cfg_definer

import wandb

def main(cfg):

    cfg_list = cfg_definer(cfg) # create configuration tree list
    for ind, cfg in enumerate(cfg_list):

        # if cfg.start_from set, it starts from the specified run
        # especially useful for failed experiments
        if hasattr(cfg, 'start_from'): 
            if (ind + 1) < cfg.start_from:
                continue
        
        print(f"RUNNING CONFIG NO {ind+1}/{len(cfg_list)}")

        # Initialize the Weights and Biases to see results more easily
        if cfg.online:  # Whether the run is saved to the online Weights and Biases
            wb_run = wandb.init(reinit=True, config=cfg.to_dict())
        else:
            wb_run = wandb.init(reinit=True, config=cfg.to_dict(), mode="disabled")
        
        run(cfg) # Runing according to the configuration
        wandb.config.update(cfg.to_dict(), allow_val_change=True) # Update the parameters
        wb_run.finish() # Finish the Weights and Biases Run

    
if __name__ == '__main__':
    
    main(cfg)

    