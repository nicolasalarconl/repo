#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors

class Graph:    
    
  
    def train_validation_loss_epoch(train_loss,validation_loss):        
        plt.figure()
        plt.plot(train_loss)
        plt.plot(validation_loss)
        plt.title('Train Loss & Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()
     
    def psnr_graph(psnr_output,psnr_dirty,psnr_diff): 
        plt.hist(psnr_output, color='b', edgecolor='b',linewidth=1,label ='clean-output')
        plt.xlabel('psnr output image')
        plt.ylabel('count images')
        plt.show()

        plt.hist(psnr_dirty,  color='g', edgecolor='g' ,linewidth=1, label = 'clean-dirty')
        plt.legend(loc='upper right')
        plt.xlabel('psnr dirty image')
        plt.ylabel('count images')
        plt.show()

        plt.hist(psnr_diff, linewidth=1,  color='r', edgecolor='r', label = 'diff output-dirty')
        plt.xlabel('diff output pnsr &  dirty pnsr')
        plt.ylabel('count images')
        plt.show() 
        
    def read_graph(path):
        img = mpimg.imread(path)
        imgplot = plt.imshow(img)
        plt.show()