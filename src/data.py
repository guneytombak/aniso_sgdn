#%%

import os
import torch
import requests

import pandas as pd
import numpy as np
from random import shuffle

from sklearn.datasets import load_iris, load_digits, fetch_california_housing
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from torch.quasirandom import SobolEngine

from src.utils import DataName
from src.model import Net

URL_ENERGY = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
URL_GRID = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv'

#%%

class Dataset(torch.utils.data.Dataset):
    """
    General dataset class that constructs the data_loader for the dataset to be used.
    """
    def __init__(self, X, y, list_IDs=None, randID=True):

        self.labels = y
        self.X = torch.Tensor(X)
  
        if list_IDs is None:
            self.list_IDs = list(range(len(self.labels)))
        else:
            self.list_IDs = list_IDs

        if randID:
            shuffle(self.list_IDs)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.X[ID]
        y = self.labels[ID]

        return X, y

def get_data(cfg):

    if cfg.dataset_name == DataName.DIGITS:
        return get_digits()
    elif cfg.dataset_name == DataName.ENERGY:
        return get_energy()
    elif cfg.dataset_name == DataName.GRID:
        return get_grid()
    elif cfg.dataset_name == DataName.HOUSE:
        return get_house()
    elif cfg.dataset_name == DataName.IRIS:
        return get_iris()
    elif cfg.dataset_name == DataName.MNIST:
        return get_mnist()
    elif isinstance(cfg.dataset_name, dict):
        return get_generated(cfg)

# Dataset loader retriever functions

def get_digits():

    digits = load_digits()
    scal = StandardScaler()
    dataset = Dataset(scal.fit_transform(digits.data), digits.target)
    return dataset
 
def get_energy():
    
    directory = './data'
    filename = 'energy.xlsx'
    
    data_path = os.path.join(directory, filename)
    
    if not os.path.isfile(data_path):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        r = requests.get(URL_ENERGY, allow_redirects=True)
        with open(data_path, 'wb') as f:
            f.write(r.content)

    xl_file = pd.ExcelFile(data_path)

    dfs = {str(ind): xl_file.parse(sheet_name) 
          for ind, sheet_name in enumerate(xl_file.sheet_names)}
    
    X = dfs['0'].iloc[: , :8].to_numpy(dtype=np.float32)
    Y = dfs['0'].iloc[: , 8:].to_numpy(dtype=np.float32)
    
    scal = StandardScaler()
    X = scal.fit_transform(X)
    Y = scal.fit_transform(Y)
    dataset = Dataset(X, Y)
    
    return dataset

def get_grid():

    directory = './data'
    filename = 'grid.csv'

    data_path = os.path.join(directory, filename)

    if not os.path.isfile(data_path):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        r = requests.get(URL_GRID, allow_redirects=True)
        with open(data_path, 'wb') as f:
            f.write(r.content)

    dfs = pd.read_csv(data_path)

    X = dfs.iloc[: , :12].to_numpy(dtype=np.float32)
    Y = dfs.iloc[: , 12].to_numpy(dtype=np.float32)

    scal = StandardScaler()
    X = scal.fit_transform(X)
    Y = np.expand_dims(Y, 1)
    Y = scal.fit_transform(Y)
    dataset = Dataset(X, Y)

    return dataset
 
def get_house():

    house = fetch_california_housing()
    scal = StandardScaler()
    Y = np.expand_dims(house.target, 1)
    X = scal.fit_transform(house.data)
    Y = scal.fit_transform(Y)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    dataset = Dataset(X,Y)

    return dataset

def get_iris():

    iris = load_iris()
    scal = StandardScaler()
    dataset = Dataset(scal.fit_transform(iris.data), iris.target)
    return dataset

def get_mnist():
    
    transform=transforms.Compose([
    #transforms.Resize((14,14)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('./data', train=False, 
                                download=True, transform=transform)
    return dataset

def get_generated(cfg):

    dataset_generator = DatasetGenerator(cfg)
    X, Y = dataset_generator.dataset
    
    dataset = Dataset(X, Y)
    return dataset
    
    
#%% 

class DatasetGenerator():
    
    def __init__(self, cfg, verbose=True):
        self.verbose = verbose
        self.model = Net(cfg=cfg, verbose=False)
        self.settings = cfg.dataset_name
        self.best_params = self._get_best_params() 

    def _get_best_params(self):
        
        rand_size = self.settings['model_par_std']
        with torch.no_grad():
            for p in self.model.parameters():
                dev = p.get_device() # get the parameter device no
                dev = dev if dev >= 0 else torch.device("cpu") # if device no is less than zero, it is cpu
            
                p_new = rand_size*torch.randn(p.shape).to(dev) # define random step vector
                p.copy_(p_new)

        return self.model.get_wb()

    @property
    def dataset(self):
        sample_size = self.settings['sample_size']
        soboleng = SobolEngine(dimension=self.settings['input_size'])
        X = 2*soboleng.draw(sample_size) - 1
        Y = self.model(X)

        X = X.clone().detach()
        Y = Y.clone().detach()

        # Addition of Gaussian Noise
        if 'noise_std' in self.settings:
            X = torch.normal(mean=X, std=self.settings['noise_std'])

        if self.verbose:
            print(f"Random dataset of size {sample_size} is created.")

        return X, Y

        