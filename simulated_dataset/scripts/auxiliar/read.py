# %%
import cupy as cp
import torch 
from astropy.io import fits


def read_pkl(path):
    return torch.load(path)

def read_fit(path,size_figure):
    hdul= fits.open(path)
    data = hdul[0].data.astype(cp.float32)
    image = cp.reshape(data,[size_figure,size_figure])
    return cp.array(image)
     

    
        
        
