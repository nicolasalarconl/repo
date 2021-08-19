# %%
import sys
from clean.listEllipses import ListEllipses
from clean.paramsEllipses import ParamsEllipses
from clean.randomImage import RandomImage
from clean.datasetImages import DatasetImages
from dirty.datasetDirty import DatasetDirty
from psf.datasetPSF import DatasetPSF


from matplotlib import pyplot as plt
from random import sample
import math


import cupy as cp
from cupyx.scipy import ndimage #as ndcupy

from auxiliar.save import save_mask,save_clean,save_dirty,save_psf
from auxiliar.psnr import get_psnr

from interferometryData import TestData
from interferometryData import TrainData
from torch.utils.data import DataLoader
from torchvision import transforms



class Dataset:
    def __init__(self,size_figure,device): 
        self.size_figure = size_figure
        self.device = device
        self.path_clean = '../dataset/data/images_'+str(size_figure)+'x'+str(size_figure)+'/images'
        self.path_psf =   '../dataset/data/images_'+str(size_figure)+'x'+str(size_figure)+'/convolutions/'
        self.path_dirty = '../dataset/data/images_'+str(size_figure)+'x'+str(size_figure)+'/convolutions/'
        
  
    def create(self,start,stop):
        params= ParamsEllipses(size_figure = self.size_figure , device = self.device)
        listEllipses = ListEllipses(params = params,index_random = start, device = self.device)

        type_psf_gauss = 'psf_gauss_'+str(self.size_figure)+'x'+str(self.size_figure)
        data_psf_gauss = DatasetPSF(size_image = self.size_figure,
                              type_psf = type_psf_gauss,
                              device = self.device
                             )
        gauss_psf = data_psf_gauss.psf_gauss(self.size_figure,self.size_figure)
        save_psf(self.path_psf,type_psf_gauss,gauss_psf)

        type_psf_real = 'psf_real_'+str(self.size_figure)+'x'+str(self.size_figure)
        data_psf_real = DatasetPSF(size_image = self.size_figure,
                              type_psf = type_psf_real,
                              device = self.device
                             )
        real_psf = data_psf_real.psf_real(self.size_figure,self.size_figure)
        save_psf(self.path_psf,type_psf_real,real_psf)

        for index in cp.arange(int(start),int(stop),1):
            clean  = RandomImage(list_figures= listEllipses,index_random =index,device = self.device)
            save_clean(self.path_clean,index,self.size_figure,clean)

            dirty_gauss = ndimage.convolve(clean.image,gauss_psf,mode='constant', cval=0.0)
            save_dirty(self.path_dirty,index,self.size_figure,type_psf_gauss,dirty_gauss)

            dirty_real = ndimage.convolve(clean.image,real_psf,mode='constant', cval=0.0)
            save_dirty(self.path_dirty,index,self.size_figure,type_psf_real,dirty_real)


    def read(self,type_psf,start,stop):
        data_image = DatasetImages(
                                   size_image = self.size_figure,
                                   device =self.device)
        data_image.read(path= self.path_clean, size_image=self.size_figure, start = start,stop = stop)

        data_dirty = DatasetDirty( size_image = self.size_figure,
                                   device =self.device,
                                   type_psf = type_psf)
        data_dirty.read(size_image = self.size_figure,
                        type_psf = type_psf,
                        start = start,
                        stop = stop,
                        path =  self.path_dirty)
        data_psf = DatasetPSF(size_image = self.size_figure,
                              type_psf = type_psf,
                              device = self.device
                             )
        data_psf.read(size_image = self.size_figure,
                      type_psf = type_psf,
                       path =  self.path_psf)
        return data_image.images,data_image.masks,data_psf.psf,data_dirty.dirtys
    
    def read_path(self,path_psf,path_clean,path_dirty,type_psf,start,stop):
        data_image = DatasetImages(
                                   size_image = self.size_figure,
                                   device =self.device)
        data_image.read(path= path_clean, size_image=self.size_figure, start = start,stop = stop)

        data_dirty = DatasetDirty( size_image = self.size_figure,
                                   device =self.device,
                                   type_psf = type_psf)
        data_dirty.read(size_image = self.size_figure,
                        type_psf = type_psf,
                        start = start,
                        stop = stop,
                        path =  path_dirty)
        data_psf = DatasetPSF(size_image = self.size_figure,
                              type_psf = type_psf,
                              device = self.device
                             )
        data_psf.read(size_image = self.size_figure,
                      type_psf = type_psf,
                       path =  path_psf)
        return data_image.images,data_image.masks,data_psf.psf,data_dirty.dirtys

    def view(self,type_psf,start,stop):
        data_clean,data_mask,psf,data_dirty = self.read(type_psf,start,stop)
        index = 1
        size = len(data_clean)
        fig = plt.figure(figsize=(4,size))
        for clean,mask,dirty in zip(data_clean,data_mask,data_dirty):
            ax = fig.add_subplot(size,4,index,xticks=[], yticks=[])
            plt.imshow(cp.asnumpy(clean))
            index = index+1
            ax = fig.add_subplot(size,4,index,xticks=[], yticks=[])
            plt.imshow(cp.asnumpy(psf))
            index = index+1
            ax = fig.add_subplot(size,4,index,xticks=[], yticks=[])
            plt.imshow(cp.asnumpy(dirty))
            index = index+1
            ax = fig.add_subplot(size,4,index, xticks=[], yticks=[])
            plt.imshow(cp.asnumpy(mask))
            index = index+1
            
            
    def view_path(self,path_psf,path_clean,path_dirty,type_psf,start,stop):
        data_clean,data_mask,psf,data_dirty = self.read_path(path_psf,path_clean,path_dirty,type_psf,start,stop)
        index = 1
        size = len(data_clean)
        fig = plt.figure(figsize=(4,size))
        for clean,mask,dirty in zip(data_clean,data_mask,data_dirty):
            ax = fig.add_subplot(size,4,index,xticks=[], yticks=[])
            plt.imshow(cp.asnumpy(clean))
            index = index+1
            ax = fig.add_subplot(size,4,index,xticks=[], yticks=[])
            plt.imshow(cp.asnumpy(psf))
            index = index+1
            ax = fig.add_subplot(size,4,index,xticks=[], yticks=[])
            plt.imshow(cp.asnumpy(dirty))
            index = index+1
            ax = fig.add_subplot(size,4,index, xticks=[], yticks=[])
            plt.imshow(cp.asnumpy(mask))
            index = index+1
        
        
    def info(self,type_psf_gauss,start,stop):
        data_clean,data_mask,psf,data_dirty = self.read(type_psf_gauss,start,stop)
        psnrs = []
        for mask,dirty in zip(data_mask,data_dirty):
            psnr_dirty = get_psnr(dirty,mask)
            psnrs.append(psnr_dirty)
        psnr_avg = cp.asnumpy(cp.average(psnrs))
        psnr_std = cp.asnumpy(cp.std(cp.array(psnrs)))
        print('PSNR AVERAGE_ '+str(psnr_avg))
        print('PSNR STD : '+str(psnr_std))       

    def tsfms(self):
        return transforms.Compose([transforms.ToTensor()])

    def train_data(self,type_psf,start,stop,perc_train,batch_train):
            size =stop-start  #size of lot of the dataset
            size_train = math.trunc(size*perc_train)
            
            data_clean,data_mask,psf,data_dirty = self.read(type_psf =type_psf,start= start,stop = start+size_train)
            trainSet = TrainData(data_dirty,data_clean,self.tsfms(), self.device)
            trainLoader=DataLoader(trainSet,batch_train,shuffle=False)
            return trainLoader
    def validate_data(self,type_psf,start,stop,perc_train,perc_validation,batch_validation):
        size =stop-start  #size of lot of the dataset
        size_validation = math.trunc(size*perc_validation)
        size_train = math.trunc(size*perc_train)
        start = start + size_train

        data_clean,data_mask,psf,data_dirty = self.read(type_psf= type_psf,start= start,stop = start+size_validation)
        validationSet = TrainData(data_dirty,data_clean,self.tsfms(), self.device)
        validationLoader=DataLoader(validationSet,batch_validation,shuffle=False)
        return validationLoader

    def test_data(self,type_psf,start,stop,perc_train,perc_validation,perc_test,batch_test):
        size = stop -start
        size_test = math.trunc(size*perc_test)
        size_train = math.trunc(size*perc_train)
        size_validation = math.trunc(size*perc_validation)
        start = start + size_train +size_validation
        data_clean,data_mask,psf,data_dirty = self.read(type_psf = type_psf,start= start,stop = start+size_validation)
   
        testSet= TestData(data_dirty,data_clean,data_mask,
                                         self.tsfms(), self.device)
        testLoader=DataLoader(testSet,batch_test,shuffle=False)
        return testLoader

    def view_data(self,data):
        index = 1
        size = len(data)
        fig = plt.figure(figsize=(4,size))
        for d in data:
            dirty,clean,mask = d               
            ax = fig.add_subplot(size,2, index, xticks=[], yticks=[])
            plt.imshow(dirty.squeeze().numpy())
            index = index+1
            ax = fig.add_subplot(size,2, index, xticks=[], yticks=[])
            plt.imshow(clean.squeeze().numpy())
            index = index+1


