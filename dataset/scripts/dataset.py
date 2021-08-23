import cupy as cp
import math
from interferometryData import TestData
from interferometryData import TrainData
from torch.utils.data import DataLoader
from torchvision import transforms
from auxiliar.read import read_fit,read_pkl


class Dataset:
    def __init__(self,devices,size_figure,path,psf,type_psf,start,stop,
                 perc_train,perc_validation,perc_test,batch_train,batch_validation,batch_test): 
        self.size_figure = size_figure
        self.devices = devices
        self.path = path
        self.psf = psf
        self.type_psf = type_psf
        self.start = start
        self.stop = stop
        self.perc_train = perc_train
        self.perc_validation= perc_validation
        self.perc_test = perc_test
        self.batch_train = batch_train
        self.batch_validation=batch_validation
        self.batch_test = batch_test
   
    def tsfms(self):
        return transforms.Compose([transforms.ToTensor()])
    
    def read_test(self,start,stop):
        cleans = []
        masks = []
        dirtys = []
        path_size = str(self.size_figure)+'x'+str(self.size_figure)
        path_clean_image = self.path+'/Clean/Clean_Images/clean_'+path_size
        path_clean_mask = self.path+'/Clean/Masks/mask_'+path_size
        path_dirty = self.path+'/'+str(self.psf)+'/Dirty_'+str(self.type_psf)+'/Dirty_'+str(self.type_psf)+'_'+path_size
        
        
        for index in cp.arange(int(start),int(stop)):
            ##Read Clean Images and Masks##
            path_index_clean = path_clean_image+'_'+str(index)+'.fits'
            path_index_mask = path_clean_mask+'_'+str(index)+'.pkl'
            clean  = read_fit(path_index_clean,self.size_figure)  
            mask =  read_pkl(path_index_mask)
            cleans.append(clean)
            masks.append(mask)
            ## Read Dirty Images ##   
            path_index_dirty = path_dirty+'_'+str(index)+'.fits'
            dirty = read_fit(path_index_dirty,self.size_figure)  
            dirtys.append(dirty)
        return cleans,masks,dirtys
    
    def read_train(self,start,stop):
    
        cleans = []
        dirtys = []
        path_size = str(self.size_figure)+'x'+str(self.size_figure)
        path_clean_image = self.path+'/Clean/Clean_Images/clean_'+path_size
        path_dirty = self.path+'/'+str(self.psf)+'/Dirty_'+str(self.type_psf)+'/Dirty_'+str(self.type_psf)+'_'+path_size
        for index in cp.arange(int(start),int(stop)):
            ##Read Clean Images##
            path_index_clean = path_clean_image+'_'+str(index)+'.fits'
            clean  = read_fit(path_index_clean,self.size_figure)  
            cleans.append(clean)
            ## Read Dirty Images##   
            path_index_dirty = path_dirty+'_'+str(index)+'.fits'
            dirty = read_fit(path_index_dirty,self.size_figure)  
            dirtys.append(dirty)
        return cleans,dirtys
  

    def train(self):
            size = (self.stop - self.start)
            size_train = math.trunc(size*self.perc_train)
            cleans,dirtys = self.read_train(0,size_train)
            ## train ###
            trainSet = TrainData(dirtys,cleans,self.tsfms(), self.devices)
            trainLoader=DataLoader(trainSet,self.batch_train,shuffle=False)
            return trainLoader
    def validation(self):
            size = (self.stop-self.start)
            size_train = math.trunc(size*self.perc_train)
            size_validation = math.trunc(size*self.perc_validation)
            start = size_train
            stop = size_train+size_validation
            cleans,dirtys = self.read_train(start = start ,stop = stop)
            ## validation ###
            validationSet = TrainData(dirtys,cleans,self.tsfms(), self.devices)
            validationLoader=DataLoader(validationSet,self.batch_validation,shuffle=False)
            return validationLoader
             
    def test(self):
        size = (self.stop-self.start)
        size_train = math.trunc(size*self.perc_train)
        size_validation = math.trunc(size*self.perc_validation)
        size_test = math.trunc(size*self.perc_test)
        start = size_train+size_validation
        stop = start+size_test
        cleans,masks,dirtys = self.read_test(start = start , stop = stop)
        #train#
        testSet= TestData(dirtys,cleans,masks,self.tsfms(), self.devices)
        testLoader=DataLoader(testSet,self.batch_test,shuffle=False)
        return testLoader

    


