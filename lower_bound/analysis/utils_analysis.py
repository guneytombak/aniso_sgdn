import numpy as np
import pickle5 as pickle
import matplotlib.pyplot as plt

def ma(x, w, type='valid'):
    y = np.zeros_like(x)
    m = np.convolve(x, np.ones(w), 'valid') / w
    if type.lower() == 'valid':
        return m
    for k in range(w//2):
        q = k+1
        y[k] = np.mean(x[0:q])
        y[-q] = np.mean(x[-q:])
        
    y[q:-k] = m
    return y

 
class Analyser():
    
    def __init__(self, filename, mak_size=None):

        self.filename = filename
        self.data = self._get_raw()
        self.mak_size = mak_size
        
        self.plt_spec = ['-r', '-b', '-k']
    
    def __getitem__(self, idx):
        
        data_i = self.data[idx]
        raw_gdl = data_i['data']['GDL']
        raw_gdl = np.reshape(np.stack(raw_gdl), (-1,3))
        gdl = list()
        if self.mak_size is None:
            mak_size = data_i['maxiter']
        else:
            mak_size = self.mak_size
        for k in range(raw_gdl.shape[1]):
            gdl.append(ma(raw_gdl[:,k], mak_size))
            
        gdl = np.stack(gdl).T
        
        return gdl
    
    def __len__(self):
        return len(self.data)
    
    def _get_raw(self):
        with open(self.filename, 'rb') as f:
            x = pickle.load(f)
        
        return x
    
    def plot(self, idx):
        gdl = self[idx]
        x = np.arange(0., gdl.shape[0], 1)
        for k in range(3):
            plt.plot(x, gdl[:,k], self.plt_spec[k])
        plt.legend(['g^2', '|g-DL|^2', 'L'], loc ="upper right")
        
        print(self.data[idx])