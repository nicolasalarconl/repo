# %%
import sqlite3
import pandas as pd
from sqlite3 import Error
import cupy as cp
import random
import pandas as pd

#https://likegeeks.com/es/tutorial-de-python-sqlite3/

# %%
class DataBase:
    def __init__(self,path):
        self.path = path
        self.con = self.sql_connection()

    def sql_connection(self):
        try:
            con = sqlite3.connect(self.path)
            return con
        except Error:
            print('Error')
    def sql_close(self):
        self.con.close()
        
    def create_table_architecture(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists architectures')
        cursorObj.execute("CREATE TABLE architectures(id INTEGER PRIMARY KEY AUTOINCREMENT,name text, number_layers integer,number_params integer)")
        self.con.commit()
              
    def create_table_params_architecture_1(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists params_architecture_1')
        cursorObj.execute("CREATE TABLE params_model(id integer PRIMARY KEY AUTOINCREMENT,k1 integer,k2 integer,k3 integer,k4 integer,k5 integer,k6 integer,k7 integer,k8 integer,p1 integer,p2 integer,p3 integer,p4 integer,s1 integer,s2 integer,s3 integer,s4 integer,out_in integer)")
        self.con.commit()   
        
    
    def create_table_model(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists models')
        cursorObj.execute("CREATE TABLE models(id INTEGER PRIMARY KEY AUTOINCREMENT,id_architecture integer,id_param integer)")
        self.con.commit()
    
    def create_table_execution(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists executions')
        cursorObj.execute("CREATE TABLE executions(id integer PRIMARY KEY AUTOINCREMENT,id_model integer, size_figure integer,type_psf integer,num_epochs integer,learning_rate integer,start integer, stop integer,len_device integer,perc_train real ,perc_validation real ,perc_test real ,batch_train integer,batch_validation integer ,batch_test integer, time_train real,time_test real,avg_psnr_output real,std_psnr_output real,avg_psnr_dirty real,std_psnr_dirty real,avg_psnr_diff real,std_psnr_diff real)")
    def create_table_losses(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists losses')
        cursorObj.execute("CREATE TABLE losses(id integer PRIMARY KEY AUTOINCREMENT,id_execution integer,loss real,type text)")
    def create_table_psnrs(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists psnrs')
        cursorObj.execute("CREATE TABLE psnrs(id integer PRIMARY KEY AUTOINCREMENT,id_execution integer,psnr real,type text)")
        self.con.commit()                      
    def create_all_tables(self):
        self.create_table_architecture()
        self.create_table_params()
        self.create_table_model()
        self.create_table_execution()
        self.create_table_losses()
        self.create_table_psnrs()
    def delete_all_tablet(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists architectures')
        cursorObj.execute('DROP table if exists params_model')
        cursorObj.execute('DROP table if exists models')
        cursorObj.execute('DROP table if exists executions')
        cursorObj.execute('DROP table if exists losses')
        cursorObj.execute('DROP table if exists psnrs')
        self.con.commit()
    def db_reset(self):
        self.delete_all_tablet()
        self.create_all_tables()
        
    def insert_architecture(self,values):
        cursorObj = self.con.cursor()
        cursorObj.execute('INSERT INTO architectures(name,number_layers,number_params) VALUES(?,?,?)', values)
        self.con.commit()
    def insert_params(self,values):
        cursorObj = self.con.cursor()
        cursorObj.execute('INSERT INTO params_model(k1,k2,k3,k4,k5,k6,k7,k8,p1,p2,p3,p4,s1,s2,s3,s4,out_in) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', values)
        self.con.commit() 
        return cursorObj.lastrowid
  
    def insert_model(self,values):
        cursorObj = self.con.cursor()
        cursorObj.execute('INSERT OR REPLACE INTO models(id_architecture, id_param) VALUES(?,?)', values)
        self.con.commit()
        return cursorObj.lastrowid
    def insert_execution(self,values):
        cursorObj = self.con.cursor()
        cursorObj.execute("INSERT INTO executions(id_model,size_figure,type_psf,num_epochs,learning_rate ,start,stop,len_device,perc_train,perc_validation,perc_test ,batch_train,batch_validation,batch_test, time_train ,time_test,avg_psnr_output,std_psnr_output,avg_psnr_dirty ,std_psnr_dirty ,avg_psnr_diff ,std_psnr_diff) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,? ,?,?)",values)
        return cursorObj.lastrowid
    
    
    def insert_losses(self,id_execution,losses,type_loss):
        cursorObj = self.con.cursor()
        for loss in losses:
            cursorObj.execute('INSERT INTO losses(id_execution,loss,type) VALUES(?,?,?)', (id_execution,loss,type_loss))
            self.con.commit()
    
    
    def update_train_time(self,id_execution,time):
            cursorObj = self.con.cursor()
            cursorObj.execute('UPDATE executions SET time_train = ? where id = ?',(time,id_execution)) 
            self.con.commit()
    def update_test_time(self,id_execution,time):
            cursorObj = self.con.cursor()
            cursorObj.execute('UPDATE executions SET time_test = ? where id = ?',(time,id_execution)) 
            self.con.commit()       
    def update_psnr_dirty(self,id_execution,avg,std):
            cursorObj = self.con.cursor()
            cursorObj.execute('UPDATE executions SET avg_psnr_dirty = ? where id = ?',(avg,id_execution)) 
            cursorObj.execute('UPDATE executions SET std_psnr_dirty = ? where id = ?',(std,id_execution)) 
            self.con.commit()
    def update_psnr_output(self,id_execution,avg,std):
            cursorObj = self.con.cursor()
            if (cp.isnan(avg) or cp.isinf(avg)  or pd.isnull(avg)):
                avg = 999
                std = 999
            cursorObj.execute('UPDATE executions SET avg_psnr_output = ? where id = ?',(avg,id_execution)) 
            cursorObj.execute('UPDATE executions SET std_psnr_output = ? where id = ?',(std,id_execution)) 
            self.con.commit()
    def update_psnr_diff(self,id_execution,avg,std):
            cursorObj = self.con.cursor()
           
            if (cp.isnan(avg) or cp.isinf(avg) or pd.isnull(avg)):
                avg = 999
                std = 999
            cursorObj.execute('UPDATE executions SET avg_psnr_diff = ? where id = ?',(avg,id_execution)) 
            cursorObj.execute('UPDATE executions SET std_psnr_diff = ? where id = ?',(std,id_execution)) 
            self.con.commit()       
   
            
    def insert_psnr(self,id_execution,psnrs,type_psnr):
        cursorObj = self.con.cursor()
        for psnr in psnrs:
            if (cp.isnan(psnr) or cp.isinf(psnr)  or pd.isnull(psnr)):
                psnr = 999
            cursorObj.execute('INSERT INTO psnrs(id_execution,psnr,type) VALUES(?,?,?)', (id_execution,psnr,type_psnr))
    
    def get_loss(self,id_execution,type_loss):
        cursorObj = self.con.cursor()
        cursorObj.execute('SELECT loss FROM losses WHERE id_execution = ? AND type = ?', (id_execution,type_loss)) 
        return cursorObj.fetchall()

    def get_psnr(self,id_execution,type_psnr):
        cursorObj = self.con.cursor()
        cursorObj.execute('SELECT psnr FROM psnrs WHERE id_execution = ? AND type = ?', (id_execution,type_psnr)) 
        psnrs = cursorObj.fetchall()
        p = []
        for psnr in psnrs:
            p.append(psnr[0])
        return p
  
    def get_execution(self,id_execution):
        cursorObj = self.con.cursor()
        cursorObj.execute('SELECT * FROM executions WHERE id ='+str(id_execution)) 
        return cursorObj.fetchall()[0]
        
        
    def exists_params(self,values):
        cursorObj = self.con.cursor()
        values =  self.random_params(size_figure)  
        cursorObj.execute('SELECT * FROM params_model WHERE k1 = ? AND k2 = ? AND k3 = ? AND k4 = ? AND k5 = ? AND k6 = ? AND k7 = ? AND k8 = ? AND p1 = ? AND p2 = ? AND p3 = ? AND p4 = ? AND s1 = ? AND s2 = ? AND s3 = ? AND s4 = ? AND out_in = ? ',values) 
        rows = cursorObj.fetchall()
        if (len(rows) == 0):
            id_param = self.insert_params(values)
            cursorObj.execute('SELECT * FROM params_model where id ='+str(id_param))
            return cursorObj.fetchall()[0]
        else: 
            return self.get_params_random(size_figure)
    def get_params(self,id_model):
        cursorObj = self.con.cursor()
        cursorObj.execute('SELECT id_param FROM models WHERE id ='+str(id_model))
        id_param = cursorObj.fetchall()
        id_param = id_param[0][0]
        cursorObj.execute('SELECT * FROM params_model WHERE id ='+str(id_param))
        return cursorObj.fetchall()[0]
        
    def random_params(self,size_figure):
        randoms  = cp.arange(1,6,1)
        randoms_out_28  = [16,32,64,128]
        randoms_out_640  = [640,220,128,64]
        k1 = random.choice(randoms)
        k2 = random.choice(randoms)
        k3 = random.choice(randoms)
        k4 = random.choice(randoms)
        k5 = random.choice(randoms)
        k6 = random.choice(randoms) 
        k7 = random.choice(randoms)
        k8 = random.choice(randoms)
        s1 = random.choice(randoms)
        s2 = random.choice(randoms)
        s3 = random.choice(randoms)
        s4 = random.choice(randoms)
        p1 = random.choice(randoms)
        p2 = random.choice(randoms)
        p3 = random.choice(randoms)
        p4 = random.choice(randoms)
        if (size_figure == 28):
              randoms_out = randoms_out_28
        if (size_figure == 640):
              randoms_out = randoms_out_640
        out_in = random.choice(randoms_out)
        values = (int(k1),int(k2),int(k3),int(k4),int(k5),int(k6),int(k7),int(k8),int(p1),int(p2),int(p3),int(p4),int(s1),int(s2),int(s3),int(s4),int(out_in))
        print(values)
        return values




