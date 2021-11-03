import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import wandb

from src.utils import seed_everything, default_config, ld2dl
from src.data import get_data
from src.model import Net, train_epoch

def run(cfg):
    
    # hyperparameters
    cfg = default_config(cfg)
    seed_everything(cfg.seed)
    
    if not hasattr(cfg, 'dev'):
        cfg.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {cfg.dev}')

    # model
    model = Net(cfg=cfg).to(cfg.dev)
    optimizer = optim.SGD(model.parameters(), cfg.lr)

    wandb.watch(model)

    if cfg.sch__use:
        scheduler = StepLR(optimizer, step_size=cfg.sch__step_size, gamma=cfg.sch__gamma)

    # data
    dataset = get_data(cfg.dataset_name)
    data_loader = torch.utils.data.DataLoader(dataset,  cfg.batch_size, shuffle=True)

    model.initialize()
    model.to(cfg.dev)
    data_list = list()
    pbar = tqdm(range(cfg.n_epochs))
    for epoch in pbar:
        data, epoch_loss = train_epoch(model, cfg, data_loader, 
                                               optimizer, epoch)
        pbar.set_description(f"Epoch Loss: {epoch_loss:.5f}", refresh=True)
        data_list.append(data)
        if cfg.sch__use:
            scheduler.step()
        
    return ld2dl(data_list)