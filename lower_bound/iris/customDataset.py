import torch
from random import shuffle

class CustomDataset(torch.utils.data.Dataset):
  
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, y, list_IDs=None, randID=True):
      'Initialization'
      
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.labels = y
      self.X = torch.Tensor(X)
      
      if list_IDs is None:
            self.list_IDs = list(range(len(self.labels)))
      else:
            self.list_IDs = list_IDs
            
      if randID:
            shuffle(self.list_IDs)
      

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.X[ID]
        y = self.labels[ID]

        return X, y