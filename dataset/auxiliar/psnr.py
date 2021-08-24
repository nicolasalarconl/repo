# %%
import cupy as cp
from matplotlib import pyplot as plt
import time
import math

# %%

def get_psnr(image,mask):
    
        std_value = cp.std(image[mask])
        reverse = cp.logical_not(mask)
        max_value = cp.max(image[reverse])
        #print('std_value: '+str(std_value))
        #print('max_value: '+str(max_value))
        psnr = cp.asnumpy((20*cp.log10(max_value/std_value))).item(0) 
        #print('psnr: '+str(psnr))
        return psnr
    
def get_psnr_torch(image,mask): 
        mask = cp.asarray(mask.squeeze(0).squeeze(0).cpu())
        image = cp.asarray(image.squeeze(0).squeeze(0).cpu().detach().numpy())
        std_value = cp.std(image[mask])
        reverse = cp.logical_not(cp.asarray(mask))
        max_value = cp.max(image[reverse])
        psnr = cp.asnumpy((20*cp.log10(max_value/std_value))).item(0) 
        return psnr
        
    
class PSNR: 
    def __init__(self,dirty,mask,device):
        self.init_device(device)
        self.get = self.get_psnr(dirty,mask) #. squeeze(),dirty.squeeze())
  
    def init_device(self,device):
         if (device  == 'cuda:0'):
            device = 0
         else:
            device = 1
         cp.cuda.Device(device).use()
            



