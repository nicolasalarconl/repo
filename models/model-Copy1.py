from tqdm import tqdm
import torch 

import sys
sys.path.append("../dataset/scripts/")
from dataset import Dataset
#from graph import Graph
#from psnr import PSNR
import time
import torch.nn as nn
import torch.optim as optim
import cupy as cp
#from d2l import torch as d2l
from auxiliar.psnr import get_psnr,get_psnr_torch
from matplotlib import pyplot as plt


class Model:
    def __init__(self,devices,size_figure,type_psf,num_epochs,learning_rate,batch_train,batch_validation,batch_test,start,stop,perc_train,perc_validation,perc_test,net):
        self.net = net
        self.devices = devices
        self.size_figure = size_figure
        self.type_psf = type_psf
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.start = start
        self.stop = stop
        self.perc_train = perc_train
        self.perc_test  = perc_test
        self.perc_validation = perc_validation
        self.batch_train = batch_train
        self.batch_validation = batch_validation
        self.batch_test = batch_test
        self.optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

 
     
    def split_batch(self,X, y):        
        return (nn.parallel.scatter(X, self.devices), nn.parallel.scatter(y, self.devices))
    
    def train_batch(self,net,X, y):
        X_shards, y_shards = self.split_batch(X, y)
        ls = [ (self.criterion(net(X_shard,device_W),y_shard.float()))
               for X_shard, y_shard, device_W in zip (X_shards, y_shards,self.devices)
             ]
        torch.cuda.synchronize()
        ls_final = ls[0]
        for l in ls[1:]:  
            ls_final = ls_final + l.item()
        return ls_final
      
                   
    def train(self,net,start,stop):
        device =  0
        data = Dataset(size_figure = self.size_figure,
                                     device = device
                      )
        trainLoader = data.train_data(self.type_psf,
                                      start,
                                      stop,
                                      self.perc_train,
                                      self.batch_train
                                     )
        validationloader = data.validate_data(self.type_psf,
                                              start,
                                              stop,
                                              self.perc_train,
                                              self.perc_validation,
                                              self.batch_validation
                                             )
        train_loss = []
        valid_loss = []
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            net.train()
            for dirty, clean in trainLoader:
                loss = self.train_batch(net,dirty,clean)
                device_final = loss.device
                net.to(device_final)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            loss = running_loss / len(trainLoader)
            train_loss.append(loss)
            #print('Epoch {} of {}, Train Loss: {:.3f}, Len Train:{}'.format(epoch+1, self.num_epochs, loss,len(trainLoader)))
            validation_loss = 0.0
            net.eval()
            size_validate = 0  
            torch.cuda.empty_cache()    
            net = net.to(self.devices[0])
            for dirty, clean in validationloader: 
                loss = self.train_batch(net,dirty,clean)
                device_final = loss.device
                net.to(device_final)
                loss.backward() 
                validation_loss += loss.item()
                torch.cuda.empty_cache()
            loss = validation_loss/len(validationloader)
            valid_loss.append(loss)
            #print('Epoch {} of {}, Validate Loss: {:.3f} Len Validate:{}'.format(epoch+1, self.num_epochs, loss,
                                                                                        #len(validationloader)))
        return net,train_loss,valid_loss
    
        
    def run_train(self,net,start,stop):     
        start_time = time.time()
        net,train_loss,valid_loss= self.train(net,start,stop)      
        stop_time = time.time()
        time_final = stop_time-start_time
        return net,train_loss,valid_loss,time_final
    
    def test(self,net,start,stop):
        device = 0
        data = Dataset(size_figure = self.size_figure,
                                     device = device
                      )        
        testLoader = data.test_data(self.type_psf,
                                      start,
                                      stop,
                                      self.perc_train,
                                      self.perc_validation,
                                      self.perc_test,
                                      self.batch_test
                                     )
        index = 0
        psnr_output = []
        psnr_dirty = []
        psnr_diff = [] 
        net.eval()
        for dirty,clean,mask in testLoader: 
            output = net(dirty,self.devices[0])
            psnr_o = get_psnr_torch(output,mask)
            psnr_d = get_psnr_torch(dirty,mask)
            psnr_df  =  psnr_o - psnr_d
            psnr_output.append(psnr_o)
            psnr_dirty.append(psnr_d)
            psnr_diff.append(psnr_df)
            index = index + 1  
        #print('Len test:{}'.format(index))
        return net,psnr_output,psnr_dirty,psnr_diff 
    
    def run_test(self,net,start,stop):
            start_time = time.time()
            net,psnr_output,psnr_dirty,psnr_diff, = self.test(net,start,stop)
            stop_time = time.time()
            time_final = stop_time-start_time
            return net,psnr_output,psnr_dirty,psnr_diff,time_final
  
    def view_test(self,net,start,stop,view_size):
        device = 0
        net.eval()
        data = Dataset(size_figure = self.size_figure,
                                     device = device
                      )        
        testLoader = data.test_data(self.type_psf,
                                      start,
                                      stop,
                                      self.perc_train,
                                      self.perc_validation,
                                      self.perc_test,
                                      self.batch_test
                                     )
        index = 1
        fig = plt.figure(figsize=(3,view_size))
        for dirty,clean,mask in testLoader: 
            output = net(dirty,self.devices[0])
            ax = fig.add_subplot(view_size,3, index, xticks=[], yticks=[])
            plt.imshow(clean.squeeze().numpy())
            index = index+1
            ax = fig.add_subplot(view_size,3, index, xticks=[], yticks=[])
            plt.imshow(dirty.squeeze().numpy())
            index = index+1
            ax = fig.add_subplot(view_size,3, index, xticks=[], yticks=[])
            plt.imshow(output.squeeze(0).squeeze(0).cpu().detach().numpy())
            index = index+1
            if (view_size*3 <= index):
                break

        

