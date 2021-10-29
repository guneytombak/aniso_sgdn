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

from src.utils import DataName

URL_ENERGY = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
URL_GRID = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv'

#%%

class Dataset(torch.utils.data.Dataset):

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

def get_data(dataset_name):

	if dataset_name == DataName.DIGITS:
		return get_digits()
	elif dataset_name == DataName.ENERGY:
		return get_energy()
	elif dataset_name == DataName.GRID:
		return get_grid()
	elif dataset_name == DataName.HOUSE:
		return get_house()
	elif dataset_name == DataName.IRIS:
		return get_iris()
	elif dataset_name == DataName.MNIST:
		return get_mnist()

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



