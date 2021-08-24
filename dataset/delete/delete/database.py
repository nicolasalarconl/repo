# %%
import sqlite3
import pandas as pd
from sqlite3 import Error
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
    def sql_create_table_model(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists models')
        cursorObj.execute("CREATE TABLE models(id INTEGER PRIMARY KEY AUTOINCREMENT,id_net integer,size_figure real,type_psf text,num_epochs integer,learning_rate real,batch_train integer,len_dataset integer)")
        self.con.commit()
    def sql_create_table_execution(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists executions')
        cursorObj.execute("CREATE TABLE executions(id integer PRIMARY KEY AUTOINCREMENT,id_model integer,len_device integer,time_train_execution real,time_test_execution real,avg_psnr real,std_psnr,avg_psnr_dirty real,std_psnr_dirty real,avg_psnr_diff real,std_psnr_diff real)")
        self.con.commit()
    def sql_create_table_model_execution(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists models_executions')
        cursorObj.execute("CREATE TABLE models_executions(id integer PRIMARY KEY AUTOINCREMENT,id_model integer,id_execution integer)")
        self.con.commit()
    def sql_create_table_losses_execution(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists losses_execution')
        cursorObj.execute("CREATE TABLE losses_execution(id integer PRIMARY KEY AUTOINCREMENT,id_execution integer,loss real,type integer)")
        self.con.commit()
    def sql_create_table_psnrs(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists psnrs')
        cursorObj.execute("CREATE TABLE psnrs(id integer PRIMARY KEY AUTOINCREMENT,id_execution integer,psnr real,type integer)")
        self.con.commit()
        
    def create_all_tables(self):
        self.sql_create_table_model()
        self.sql_create_table_execution()
        self.sql_create_table_model_execution()
        self.sql_create_table_losses_execution()
        self.sql_create_table_psnrs()
    def delete_all_tablet(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists models')
        cursorObj.execute('DROP table if exists executions')
        cursorObj.execute('DROP table if exists models_executions')
        cursorObj.execute('DROP table if exists losses_execution')
        cursorObj.execute('DROP table if exists psnrs')
        self.con.commit()
        
    def sql_insert_model(self,values):
        cursorObj = self.con.cursor()
        cursorObj.execute('INSERT OR REPLACE INTO models(id,id_net, size_figure,type_psf,num_epochs,learning_rate,batch_train,len_dataset) VALUES(?,?,?,?,?,?,?,?)', values)
        self.con.commit()
    def sql_insert_executions(self,values):
            cursorObj = self.con.cursor()
            cursorObj.execute('INSERT OR REPLACE INTO executions(id,id_model,len_device, time_train_execution,time_test_execution,avg_psnr,std_psnr,avg_psnr_dirty,std_psnr_dirty,avg_psnr_diff,std_psnr_diff) VALUES(?,?,?,?,?,?,?,?,?,?,?)', values)
            self.con.commit()
    def sql_insert_losses(self,values):
            cursorObj = self.con.cursor()
            cursorObj.execute('INSERT INTO losses_execution(id_execution,loss,type) VALUES(?,?,?)', values)
            self.con.commit()
            
    def sql_update_train_time(self,id_execution,value):
            cursorObj = self.con.cursor()
            cursorObj.execute('UPDATE executions SET time_train_execution = ? where id = ?',(value,id_execution)) 
            self.con.commit()
    def sql_insert_psnr(self,values):
            cursorObj = self.con.cursor()
            cursorObj.execute('INSERT INTO psnrs(id_execution,psnr,type) VALUES(?,?,?)', values)
            self.con.commit() 
       
    def sql_update_test_time(self,id_execution,value):
            cursorObj = self.con.cursor()
            cursorObj.execute('UPDATE executions SET time_test_execution = ? where id = ?',(value,id_execution)) 
            self.con.commit()
    def sql_update_psnr_clean(self,id_execution,avg,std):
            cursorObj = self.con.cursor()
            cursorObj.execute('UPDATE executions SET avg_psnr = ? where id = ?',(avg,id_execution)) 
            cursorObj.execute('UPDATE executions SET std_psnr = ? where id = ?',(std,id_execution)) 
            self.con.commit()
    def sql_update_psnr_dirty(self,id_execution,avg,std):
            cursorObj = self.con.cursor()
            cursorObj.execute('UPDATE executions SET avg_psnr_dirty = ? where id = ?',(avg,id_execution)) 
            cursorObj.execute('UPDATE executions SET std_psnr_dirty = ? where id = ?',(std,id_execution)) 
            self.con.commit()
    def sql_update_psnr_diff(self,id_execution,avg,std):
            cursorObj = self.con.cursor()
            cursorObj.execute('UPDATE executions SET avg_psnr_diff = ? where id = ?',(avg,id_execution)) 
            cursorObj.execute('UPDATE executions SET std_psnr_diff = ? where id = ?',(std,id_execution)) 
            self.con.commit()
        
 
     
        
    def sql_fetch_condition(self,con):
        cursorObj = self.con.cursor()

        

      
