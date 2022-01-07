import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import wandb
from datetime import datetime

from src.utils import seed_everything, default_config
from src.data import get_data
from src.model import Net
from src.training import train_epoch

def run(cfg):

    seed_everything(cfg.seed)   # Seeding everything for reproducibility
    cfg = default_config(cfg)   # Extracting default hyperparameters specific to the dataset

    run_name = cfg.experiment_name + '_' + wandb.run.id
    wandb.run.name = run_name
    
    if not hasattr(cfg, 'dev'):
        cfg.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {cfg.dev}')

    # model
    model = Net(cfg=cfg).to(cfg.dev)
    optimizer = optim.SGD(model.parameters(), cfg.lr)

    if cfg.sch__use:
        scheduler = StepLR(optimizer, step_size=cfg.sch__step_size, gamma=cfg.sch__gamma)

    # data
    dataset = get_data(cfg)
    batch_loader = torch.utils.data.DataLoader(dataset,  cfg.batch_size, shuffle=True)
    single_loader = torch.utils.data.DataLoader(dataset, 1, shuffle=False) #! Important

    wandb.watch(model, log="all", log_graph=(True)) # watch the model parameters

    model.initialize() # initialize the model
    model.to(cfg.dev) # put model into the device (cuda or cpu)
    pbar = tqdm(range(cfg.n_epochs)) # progress bar
    for epoch in pbar:
        epoch_loss = train_epoch(model, cfg, batch_loader, single_loader, optimizer)
        pbar.set_description(f"Epoch Loss: {epoch_loss:.5f}", refresh=True) # update progress bar
        if cfg.sch__use:
            scheduler.step()
