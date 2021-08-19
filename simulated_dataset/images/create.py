# %%
# %%
import sys
sys.path.append("../scripts")

from clean.listEllipses import ListEllipses
from clean.paramsEllipses import ParamsEllipses
from clean.randomImage import RandomImage
from clean.datasetImages import DatasetImages
#from dirty.datasetDirty import DatasetDirty
#from psf.datasetPSF import DatasetPSF


#from matplotlib import pyplot as plt
#from random import sample
#import math


import cupy as cp
from cupyx.scipy import ndimage #as ndcupy

from auxiliar.save import save_mask,save_clean,save_dirty,save_psf
from auxiliar.read import read_fit

#from auxiliar.psnr import get_psnr

#from interferometryData import TestData
#from interferometryData import TrainData
#from torch.utils.data import DataLoader
#from torchvision import transforms

from tqdm import tqdm

DEVICE = int(sys.argv[1])
SIZE_FIGURE = int(sys.argv[2])
ACTION = str(sys.argv[3])
START = int(sys.argv[4]) 
STOP   = int(sys.argv[5]) 
if(len(sys.argv) == 7): TYPE_PSF = str(sys.argv[6])
class Create:
    def __init__(self,device,size_figure): 
        self.size_figure = size_figure
        self.device = self.init_device(device)
        self.path_clean = './'+str(size_figure)+'/Clean'
        self.path_dirty = './'+str(size_figure)
        self.path_psf = './PSF'
    def init_device(self,device):
        cp.cuda.Device(device).use()
        return device
    
    def clean_images(self,start,stop):
        params= ParamsEllipses(size_figure = self.size_figure , device = self.device)
        listEllipses = ListEllipses(params = params,index_random = start, device = self.device)
        for index in tqdm( cp.arange(int(start),int(stop),1),leave=True):
            clean  = RandomImage(list_figures= listEllipses,index_random =index,device = self.device)
            save_clean(self.path_clean,index,self.size_figure,clean)
            
    def dirty_images(self,start,stop,type_psf):
        psf_gauss = read_fit(path =self.path_psf+'/'+type_psf+'/PSF_Gauss/psf_gauss.fits',size_figure = self.size_figure)
        psf_real =  read_fit(path =self.path_psf+'/'+type_psf+'/PSF_Real/psf_real.fits',size_figure = self.size_figure)
        path_clean_image = self.path_clean+'/Clean_Images/'+'clean_'+str(self.size_figure)+'x'+str(self.size_figure)
        for index in tqdm(cp.arange(int(start),int(stop),1),leave=True):
            path_clean_image_index = path_clean_image+'_'+str(index)+'.fits'
            clean_image =  read_fit(path = path_clean_image_index,size_figure = self.size_figure)
            dirty_image_gauss = ndimage.convolve(clean_image,psf_gauss,mode='constant', cval=0.0)
            dirty_image_real = ndimage.convolve(clean_image,psf_real,mode='constant', cval=0.0)
            path_dirty_image = self.path_dirty+'/'+type_psf
            save_dirty(path_dirty_image,index,self.size_figure,'Dirty_Gauss',dirty_image_gauss)
            save_dirty(path_dirty_image,index,self.size_figure,'Dirty_Real',dirty_image_real)

            
if __name__ == '__main__':
    cr = Create(DEVICE,SIZE_FIGURE)
    if (ACTION == 'clean_images'):
        print('creating CLEAN IMAGES....')
        cr.clean_images(START,STOP)
        print('Finished')
    if (ACTION == 'dirty_images'):
        print('creating DIRTY IMAGES....')
        cr.dirty_images(START,STOP,TYPE_PSF)
        print('Finished')

 
        