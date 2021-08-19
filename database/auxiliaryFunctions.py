# %%
import torch 
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import cv2
from astropy.io import fits
import cupy as cp
import pandas as pd
import numpy as np

from database import DataBase

#import nvidia_smi


# %%
class AuxiliaryFunctions:
    def make_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
    def write_pkl(path,data):
         torch.save(data, path)
            

       
    '''def size_model(net):
        mem_params = sum([param.nelement()*param.element_size() for param in net.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in net.buffers()])
        mem = mem_params + mem_bufs # in bytes
        return mem*1e-6'''
        
    
    '''def save_fit_image(img, path):
        hdu_image =fits.PrimaryHDU(cp.asnumpy(img))
        hdu_image.writeto(path,overwrite=True)
        
    def display_fit_image(title,path):
        hdul=fits.open(path)
        data = hdul[0].data.astype(cp.float32)
        size = data.shape[2]
        image = cp.reshape(data,[size,size])
        plt.imshow(image)
        plt.show()

    def get_device(id):
        if torch.cuda.is_available():
            device = 'cuda:'+str(id)
        else:
            device = 'cpu'
        print('device: '+str(device))
        return device
    
    def load_mode(path,file = None):
        if ( file is None):
            return torch.load(path+'/model.pkl')
        else:
            return torch.load(path+file)
        
    def save_model(model,path,file = None):
        if (file is None):
            torch.save(model, path+'/model.pkl')
        else:
            torch.save(model, path+file)

            
    def display(a, title1 = "Original"):
        plt.imshow(a), plt.title(title1)
        plt.show()'''
    
       
    '''def db_reset(path_database):
        db = DataBase(path_database)
        db.delete_all_tablet()
        db.create_all_tables()
        db.sql_close()
    def db_new_model(path_database,values):
        db = DataBase(path_database)
        db.sql_insert_model(values)
        db.sql_close()
            
            
    def db_loss(path_database,db_model):
        db = DataBase(path_database)
        db.sql_insert_model(values)
        db.sql_close()
            
    def db_valid(path_database,db_model):
        db = DataBase(path_database)
        db.sql_insert_model(values)
        db.sql_close()

        
    def db_new_executions(path_database,values):
        db = DataBase(path_database)
        db.sql_insert_executions(values)
        db.sql_close()
        
    def db_train_loss(path_database,id_execution,losses):
        db = DataBase(path_database)
        for loss in losses:
            db.sql_insert_losses((id_execution,loss,0))
        db.sql_close()
    def db_validate_loss(path_database,id_execution,losses):
        db = DataBase(path_database)
        for loss in losses:
            db.sql_insert_losses((id_execution,loss,1))
        db.sql_close()
        
    def db_train_time(path_database,id_execution,value):
        db = DataBase(path_database)
        db.sql_close()   
    
    def db_psnr_dirty(path_database,id_execution,psnrs):
        db = DataBase(path_database)
        avg_psnr = cp.asnumpy(cp.average(psnrs)).item()
        std_psnr = cp.asnumpy(cp.std(cp.array(psnrs))).item()
        db.sql_update_psnr_dirty(id_execution,avg_psnr,std_psnr)
        for psnr in psnrs:
            db.sql_insert_psnr((id_execution,psnr,0))
        db.sql_close() 
        
    def db_psnr_clean(path_database,id_execution,psnrs):
        db = DataBase(path_database)
        avg_psnr = cp.asnumpy(cp.average(psnrs)).item()
        std_psnr = cp.asnumpy(cp.std(cp.array(psnrs))).item()
      
        db.sql_update_psnr_clean(id_execution,avg_psnr,std_psnr)

        for psnr in psnrs:
            db.sql_insert_psnr((id_execution,psnr,1))
        db.sql_close()    
   
    def db_psnr_diff(path_database,id_execution,psnrs):
        db = DataBase(path_database)
        avg_psnr = cp.asnumpy(cp.average(psnrs)).item()
        std_psnr = cp.asnumpy(cp.std(cp.array(psnrs))).item()
        db.sql_update_psnr_diff(id_execution,avg_psnr,std_psnr)
        db = DataBase(path_database)
        for psnr in psnrs:
            db.sql_insert_psnr((id_execution,psnr,2))
        db.sql_close() 
        
        
    def db_test_time(path_database,id_execution,value):
        db = DataBase(path_database)
        db.sql_update_test_time(id_execution,value)
        db.sql_close()   
    
    
    def write_log(path,data):
        file1 = open(path,"w+")
        file1.writelines(data)
        file1.close() 
        
    def write_statistic(path,psnr_diff,psnr_dirty,psnr_output):
        path = path+'/statistic.txt'
        file = open(path,"w+")
        avg_psnr_diff = cp.asnumpy(cp.average(psnr_diff))
        avg_psnr = cp.asnumpy(cp.average(psnr_output))
        avg_psnr_dirty = cp.asnumpy(cp.average(psnr_dirty))
        std_psnr_diff = cp.asnumpy(cp.std(cp.array(psnr_diff)))
       
        std_psnr_dirty = cp.asnumpy(cp.std(cp.array(psnr_dirty)))
        file.writelines(['avg_diff: ' ,str(avg_psnr_diff),', std_diff: ' ,str(std_psnr_diff),', avg: ' ,str(avg_psnr),', std: ' ,str(std_psnr),', avg_dirty: ' ,str(avg_psnr_dirty),', std_dirty: ' ,str(std_psnr_dirty),'\n'])
    def read_statistic(path):
        path = path+'/statistic.txt'
        f = open(path, "r")
        r = f.read()
        f.close()
        return r

    def send_alert(path_log,psnr_diff,psnr_dirty,psnr_output,data_model):
        path_avg = path_log+'/avg.csv'
        path_alert = path_log+'/alert.txt'
        avg_psnr_diff = cp.asnumpy(cp.average(psnr_diff))
        avg_psnr = cp.asnumpy(cp.average(psnr_output))
        avg_psnr_dirty = cp.asnumpy(cp.average(psnr_dirty))
        std_psnr_diff = cp.asnumpy(cp.std(cp.array(psnr_diff)))
        std_psnr = cp.asnumpy(cp.std(cp.array(psnr_output)))
        std_psnr_dirty = cp.asnumpy(cp.std(cp.array(psnr_dirty)))

        if not os.path.isfile(path_avg):
            #df = pd.DataFrame([avg_psnr])
            #df.to_csv(path_avg,index=False)
            file = open(path_alert,"w+")
            file.writelines(['model: '+data_model,', avg diff: ' ,str(avg_psnr_diff),', std diff: ' ,str(std_psnr_diff),
                                 ', avg: ' ,str(avg_psnr),', std: ' ,str(std_psnr),', avg_dirty: ' ,str(avg_psnr_dirty),
                                 ', std_dirty: ' ,str(std_psnr_dirty),'\n'])
            file.close()
        else:
            avg = pd.read_csv(path_avg).to_numpy()
            maximum = max(avg)[0]
            if(avg_psnr > maximum):
                #df = pd.DataFrame([avg_psnr])
                #df.to_csv(path_avg,index=False)
                file = open(path_alert,"a")
                file.writelines(['model: '+data_model,', avg diff: ' ,str(avg_psnr_diff),', std diff: ' ,str(std_psnr_diff),
                                 ', avg: ' ,str(avg_psnr),', std: ' ,str(std_psnr),', avg_dirty: ' ,str(avg_psnr_dirty),
                                 ', std_dirty: ' ,str(std_psnr_dirty),'\n'])
                file.close()
                    
             
            
    def log(path_log,data_model,status,time = None):
        path_log = path_log+'/log.txt'
        if not os.path.isfile(path_log):
            file = open(path_log,"a+")
        else:
            file = open(path_log,"a")
        if (time == None):
            file.writelines(['model: '+data_model,'| status:' ,status,'\n'])
        else:
            file.writelines(['model: '+data_model,'| status:' ,status, '| time:',str(round(time, 3)),'\n'])
        file.close()'''
        
                           
        
    