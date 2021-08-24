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
        #self.create_table_params()
        #self.create_table_model()
        #self.create_table_execution()
        #self.create_table_losses()
        #self.create_table_psnrs()
    def delete_all_tablet(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists architectures')
        cursorObj.execute('DROP table if exists params_model')
        cursorObj.execute('DROP table if exists models')
        cursorObj.execute('DROP table if exists executions')
        cursorObj.execute('DROP table if exists losses')
        cursorObj.execute('DROP table if exists psnrs')
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
        cursorObj.execute('INSERT INTO params(k1,k2,k3,k4,k5,k6,k7,k8,k9,p1,p2,p3,p4,p5,s1,s2,s3,s4,out_in) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', values)
        self.con.commit() 
        return cursorObj.lastrowid
        
    def exists_params(self,values):
        cursorObj = self.con.cursor()
        cursorObj.execute('SELECT * FROM params WHERE k1 = ? AND k2 = ? AND k3 = ? AND k4 = ? AND k5 = ? AND k6 = ? AND k7 = ? AND k8 = ? AND p1 = ? AND p2 = ? AND p3 = ? AND p4 = ? AND s1 = ? AND s2 = ? AND s3 = ? AND s4 = ? AND out_in = ? ',values) 
        rows = cursorObj.fetchall()
        if (len(rows) == 0):
            return False
            ##id_param = self.insert_params(values)
            ##cursorObj.execute('SELECT * FROM params_model where id ='+str(id_param))
            return cursorObj.fetchall()[0]
        else: 
            return True 
        ###self.get_params_random(size_figure)
    ### MODEL ########
    
    def create_table_model(self):
        cursorObj = self.con.cursor()
        cursorObj.execute('DROP table if exists models')
        cursorObj.execute("CREATE TABLE models(id INTEGER PRIMARY KEY AUTOINCREMENT,id_param integer,execute integer)")
        self.con.commit()
        
    def insert_model(self,values):
        cursorObj = self.con.cursor()
        cursorObj.execute('INSERT OR REPLACE INTO models(id_param,execute) VALUES(?,?)', values)
        self.con.commit()
        return cursorObj.lastrowid


