import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch.utils.tensorboard import SummaryWriter

from utils import seed_everything, default_config, ld2dl
from data import get_data
from model import Net, train_epoch

def run(cfg):
    
    # hyperparameters
    cfg = default_config(cfg)
    sch = cfg.sch
    seed_everything(cfg.seed)

    writer = SummaryWriter(f'./results/{cfg.name}/{cfg.date}')
    
    if not hasattr(cfg, 'dev'):
        cfg.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = Net(cfg=cfg).to(cfg.dev)
    optimizer = optim.SGD(model.parameters(), cfg.lr)

    if sch.use:
        scheduler = StepLR(optimizer, step_size=sch.step_size, gamma=sch.gamma)

    # data
    dataset = get_data(cfg.dataset_name)
    data_loader = torch.utils.data.DataLoader(dataset,  cfg.batch_size, shuffle=True)

    model.initialize()
    model.to(cfg.dev)
    data_list = list()
    for epoch in range(cfg.n_epochs):
        data, writer, epoch_loss = train_epoch(model, cfg, data_loader, 
                                               optimizer, epoch, writer)
        data_list.append(data)
        if sch.use:
            scheduler.step()
        
    #writer.add_hparams(cfg.to_dict(), {'loss': epoch_loss})
        
    writer.close()

    return ld2dl(data_list)