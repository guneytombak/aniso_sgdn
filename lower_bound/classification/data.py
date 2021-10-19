import torch
from random import shuffle
from sklearn.datasets import load_iris
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler

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

	dataset = datasets.MNIST('../data', train=False, 
								download=True, transform=transform)
	return dataset

def get_data(dataset_name):

	if dataset_name.lower() == 'mnist':
		return get_mnist()
	elif dataset_name.lower() == 'iris':
		return get_iris()