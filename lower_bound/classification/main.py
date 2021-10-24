from config import cfg
from run import run
from utils import cfg_definer
import pickle5 as pickle
from datetime import datetime

cfgs_fin = list()

def main(cfg):
    
    cfg_list = cfg_definer(cfg)
    for cfg in cfg_list:
        cfg.data = run(cfg)
        cfg.finish_date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        with open(f'./results/{cfg.name}/{cfg.finish_date}.pickle', 'wb') as handle:
            pickle.dump(cfg.to_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
            cfgs_fin.append(cfg.to_dict())
    
    with open(f'./results/{cfg.name}.pickle', 'wb') as handle:
        pickle.dump(cfgs_fin, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
if __name__ == '__main__':
    
    main(cfg)