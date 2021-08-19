# %%
from astropy.io import fits
import cupy as cp
import os
import torch 

def make_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
            
def save_pkl(path,data):
         torch.save(data, path)
        
def save_fit(path,image):        
        hdu_image =fits.PrimaryHDU(cp.asnumpy(image))  
        hdu_image.writeto(path,overwrite=True)
        
def save_mask(path,mask):
        save_pkl(path,mask) 
        
def save_clean(path,index,size_figure,clean):
        path_clean = path+'/Clean_Images'
        path_mask = path+'/Masks'
        make_dir(path_clean)
        make_dir(path_mask)
        path_clean_index = path_clean+'/clean_'+str(size_figure)+'x'+str(size_figure)+'_'+str(index)+'.fits'
        path_mask_index = path_mask+'/mask_'+str(size_figure)+'x'+str(size_figure)+'_'+str(index)+'.pkl'
        save_fit(path_clean_index,clean.image)
        save_mask(path_mask_index,clean.mask)
        
def save_dirty(path,index,size_figure,type_dirty,dirty):
        path = path+'/'+str(type_dirty)
        make_dir(path)   
        path_index = path+'/'+str(type_dirty)+'_'+str(size_figure)+'x'+str(size_figure)+'_'+str(index)+'.fits'
        save_fit(path_index,dirty)

def save_psf(path,type_psf,psf):
        make_dir(path)   
        path = path+'/'+str(type_psf)+'.fits'
        save_fit(path,psf)
        
        


           