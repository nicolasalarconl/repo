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
    ### CONFIG ### 
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

    def create_all_tables(self):
        self.create_table_architecture()
        self.create_table_param()
        self.create_table_model()
        self.create_table_dataset()
        self.create_table_hyperparameter()
        self.create_table_execution()
        self.create_table_loss()
        self.create_table_psnr()
        self.create_table_result()
        
    def delete_all_tablet(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists architectures')
        cursorObj.execute('DROP table if exists params')
        cursorObj.execute('DROP table if exists models')
        cursorObj.execute('DROP table if exists datasets')
        cursorObj.execute('DROP table if exists hyperparameters')
        cursorObj.execute('DROP table if exists executions')
        cursorObj.execute('DROP table if exists losses')
        cursorObj.execute('DROP table if exists psnrs')
        cursorObj.execute('DROP table if exists results')
        self.con.commit()
    def reset(self):
        self.delete_all_tablet()
        self.create_all_tables()
        
        
        
    ### ARCHITECTURE ####
    def create_table_architecture(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists architectures')
        cursorObj.execute("CREATE TABLE architectures(id INTEGER PRIMARY KEY AUTOINCREMENT,name text, number_layers integer,number_params integer)")
        self.con.commit()     
    def insert_architecture(self,values):
        cursorObj = self.con.cursor()
        cursorObj.execute('INSERT INTO architectures(name,number_layers,number_params) VALUES(?,?,?)', values)
        self.con.commit()
        return cursorObj.lastrowid
    def get_all_architectures(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('SELECT * FROM architectures') 
        return cursorObj.fetchall()
  
    ### PARAMS #######
    def create_table_param(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists params')
        cursorObj.execute("CREATE TABLE params(id integer PRIMARY KEY AUTOINCREMENT,k1 integer,k2 integer,k3 integer,k4 integer,k5 integer,k6 integer,k7 integer,k8 integer,k9 integer,p1 integer,p2 integer,p3 integer,p4 integer,p5 integer,s1 integer,s2 integer,s3 integer,s4 integer,out_in integer)")
        self.con.commit()  
        
    def insert_params(self,values):
        cursorObj = self.con.cursor()
        cursorObj.execute('INSERT INTO params(k1,k2,k3,k4,k5,k6,k7,k8,k9,p1,p2,p3,p4,p5,s1,s2,s3,s4,out_in) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', values)
        self.con.commit() 
        return cursorObj.lastrowid
    
    
    def get_all_params(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('SELECT * FROM params') 
        return cursorObj.fetchall()
    
        
    def exists_params(self,values):
        cursorObj = self.con.cursor()
        cursorObj.execute('SELECT * FROM params WHERE k1 = ? AND k2 = ? AND k3 = ? AND k4 = ? AND k5 = ? AND k6 = ? AND k7 = ? AND k8 = ? AND k9 = ? AND p1 = ? AND p2 = ? AND p3 = ? AND p4 = ? AND p5 = ? AND s1 = ? AND s2 = ? AND s3 = ? AND s4 = ? AND out_in = ? ',values) 
        rows = cursorObj.fetchall()
        if (len(rows) == 0):
            return False
        else: 
            return True 
    
    ### MODEL ########
 
    def create_table_model(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists models')
        cursorObj.execute("CREATE TABLE models(id INTEGER PRIMARY KEY AUTOINCREMENT,id_param integer,finish integer)")
        self.con.commit()
        
    def insert_model(self,values):
        cursorObj = self.con.cursor()
        cursorObj.execute('INSERT OR REPLACE INTO models(id_param,finish) VALUES(?,?)', values)
        self.con.commit()
        return cursorObj.lastrowid
    def get_all_models(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('SELECT * FROM models') 
        return cursorObj.fetchall()
    
    def update_model_finish(self,id_model,finish):
        cursorObj = self.con.cursor()
        cursorObj.execute('UPDATE models SET finish = ? where id = ?',(finish,id_model)) 
        self.con.commit()
    
      ### Dataset ####
        
 
    def create_table_dataset(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists datasets')
        cursorObj.execute("CREATE TABLE datasets(id INTEGER PRIMARY KEY AUTOINCREMENT,start integer, stop integer,psf text, type_psf text)")
        self.con.commit()     
    def insert_dataset(self,values):
        cursorObj = self.con.cursor()
        cursorObj.execute('INSERT INTO datasets(start,stop,psf,type_psf) VALUES(?,?,?,?)', values)
        self.con.commit()
        return cursorObj.lastrowid
  
    def get_all_datasets(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('SELECT * FROM datasets') 
        return cursorObj.fetchall()
    
    
    ##HYPERPARAMETER###
    
    def create_table_hyperparameter(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists hyperparameters')
        cursorObj.execute("CREATE TABLE hyperparameters(id INTEGER PRIMARY KEY AUTOINCREMENT,num_epochs integer, learning_rate real, perc_train real,batch_train integer,perc_validation real,batch_validation integer,perc_test real,batch_test integer)")
        self.con.commit()     
    def insert_hyperparameter(self,values):
        cursorObj = self.con.cursor()
        cursorObj.execute('INSERT INTO hyperparameters(num_epochs,learning_rate,perc_train,batch_train,perc_validation,batch_validation,perc_test,batch_test) VALUES(?,?,?,?,?,?,?,?)', values)
        self.con.commit()
        return cursorObj.lastrowid
    
    def get_all_hyperparameters(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('SELECT * FROM hyperparameters') 
        return cursorObj.fetchall()
    
    
    ### EXECUTIONS ####
    def create_table_execution(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists executions')
        cursorObj.execute("CREATE TABLE executions(id INTEGER PRIMARY KEY AUTOINCREMENT,id_model integer,id_hyperparameter integer,id_dataset integer,finish integer)")
        self.con.commit()  
        
    def insert_execution(self,values):
        cursorObj = self.con.cursor()
        cursorObj.execute('INSERT INTO executions(id_model,id_hyperparameter,id_dataset,finish) VALUES(?,?,?,?)', values)
        self.con.commit()
        return cursorObj.lastrowid

    def get_all_executions(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('SELECT * FROM executions ') 
        return cursorObj.fetchall()
  
    def update_execution_finish(self,id_execution,finish):
        cursorObj = self.con.cursor()
        cursorObj.execute('UPDATE executions SET finish = ? where id = ?',(finish,id_execution)) 
        self.con.commit()
      
    
    ### LOSSES ## 
    
    def create_table_loss(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists losses')
        cursorObj.execute("CREATE TABLE losses(id integer PRIMARY KEY AUTOINCREMENT,id_execution integer,loss real,type text)")
        self.con.commit()
        
        
    def insert_loss(self,values):
            cursorObj = self.con.cursor()
            cursorObj.execute('INSERT INTO losses(id_execution,loss,type) VALUES(?,?,?)', values)
            self.con.commit()
            return cursorObj.fetchall()
        
    def get_all_losses(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('SELECT * FROM losses') 
        return cursorObj.fetchall()
        
     
    ### PSNRS ## 
    
    def create_table_psnr(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists psnrs')
        cursorObj.execute("CREATE TABLE psnrs(id integer PRIMARY KEY AUTOINCREMENT,id_execution integer,psnr real,type text)")
        self.con.commit()
        
        
    def insert_psnr(self,values):
            cursorObj = self.con.cursor()
            cursorObj.execute('INSERT INTO psnrs(id_execution,psnr,type) VALUES(?,?,?)', values)
            self.con.commit()
            return cursorObj.fetchall()
        
    def get_all_psnrs(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('SELECT * FROM psnrs') 
        return cursorObj.fetchall()
            
     #RESULTS###
    
    def create_table_result(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists results')
        cursorObj.execute("CREATE TABLE results(id integer PRIMARY KEY AUTOINCREMENT,id_execution integer,len_device integer,time_train real,time_test real,memory_train real, memory_test real,avg_psnr_output real,std_psnr_output real,avg_psnr_dirty real,std_psnr_dirty real,avg_psnr_diff real,std_psnr_diff real)")
        self.con.commit()
       
    def insert_result(self,values):
        cursorObj = self.con.cursor()
        cursorObj.execute('INSERT INTO results(id_execution,len_device,time_train,time_test,memory_train,memory_test,avg_psnr_output ,std_psnr_output,avg_psnr_dirty,std_psnr_dirty,avg_psnr_diff,std_psnr_diff) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)', values)
        self.con.commit()
        return cursorObj.lastrowid
    
    def get_all_results(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('SELECT * FROM results') 
        return cursorObj.fetchall()
    
     