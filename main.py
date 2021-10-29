import pickle5 as pickle
from datetime import datetime

from config import cfg
from src.run import run
from src.utils import cfg_definer

cfgs_fin = list()

def main(cfg):
    
    cfg_list = cfg_definer(cfg)
    n_cfgs = len(cfg_list)
    for ind, cfg in enumerate(cfg_list):
        print(f"RUNNING CONFIG NO {ind+1}/{n_cfgs}")
        cfg.data = run(cfg)
        cfg.finish_date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        with open(f'./results/{cfg.name}/{cfg.finish_date}.pickle', 'wb') as handle:
            pickle.dump(cfg.to_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
            cfgs_fin.append(cfg.to_dict())
    
    with open(f'./results/{cfg.name}.pickle', 'wb') as handle:
        pickle.dump(cfgs_fin, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
if __name__ == '__main__':
    
    main(cfg)