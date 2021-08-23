from tqdm import tqdm
import torch 
import sys
sys.path.append("../dataset/scripts/")
from dataset import Dataset
import time
import torch.nn as nn
import torch.optim as optim
import cupy as cp
from auxiliar.psnr import get_psnr,get_psnr_torch
from matplotlib import pyplot as plt


class Model:
    def __init__(self,devices,num_epochs,learning_rate,net):
        self.net = net
        self.devices = devices
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
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
      
                   
    def train(self,net,trainLoader,validationloader):
        memory = 0 
        start_time = time.time()
        train_loss = []
        valid_loss = []
        for epoch in tqdm(range(self.num_epochs)):
            running_loss = 0.0
            net.train()
            for dirty,clean in trainLoader:
                loss = self.train_batch(net,dirty,clean)
                device_final = loss.device
                net.to(device_final)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            loss = running_loss / len(trainLoader)
            train_loss.append(loss)
            print('Epoch {} of {}, Train Loss: {:.3f}, Len Train:{}'.format(epoch+1, self.num_epochs, loss,len(trainLoader)))
            validation_loss = 0.0
            net.eval()
            torch.cuda.empty_cache()    
            net = net.to(self.devices[0])
            for dirty,clean in validationloader: 
                loss = self.train_batch(net,dirty,clean)
                device_final = loss.device
                net.to(device_final)
                loss.backward() 
                validation_loss += loss.item()
                torch.cuda.empty_cache()
            loss = validation_loss/len(validationloader)
            valid_loss.append(loss)
            print('Epoch {} of {}, Validate Loss: {:.3f} Len Validate:{}'.format(epoch+1, self.num_epochs, loss,
                                                                                        len(validationloader)))
                
        stop_time = time.time()
        time_final = stop_time-start_time
        return net,train_loss,valid_loss,time_final,memory 
        


    def test(self,net,testLoader):
        memory = 0
        start_time = time.time()
        psnr_output = []
        psnr_dirty = []
        psnr_diff = [] 
        net.eval()
        for dirty,clean,mask in tqdm(testLoader): 
            output = net(dirty,self.devices[0])
            psnr_o = get_psnr_torch(output,mask)
            psnr_d = get_psnr_torch(dirty,mask)
            psnr_df  =  psnr_o - psnr_d
            psnr_output.append(psnr_o)
            psnr_dirty.append(psnr_d)
            psnr_diff.append(psnr_df)
        stop_time = time.time()
        time_final = stop_time-start_time
        return net,psnr_output,psnr_dirty,psnr_diff,time_final,memory
    
   
 
        

