# %%
# %%
import os
import sys
sys.path.append("../database")
from database import DataBase
PATH_DB = '../database/database.db'
NAME =  str(sys.argv[1]) 
NUMBER_LAYERS = int(sys.argv[2])
NUMBER_PARAMS = int(sys.argv[3])


            
class Create:
    def __init__(self,name,number_layers,number_params): 
        db = DataBase(PATH_DB)
        values = (name,number_layers,number_params)
        db.insert_architecture(values)
        db.sql_close()
        if not os.path.exists(name):
            os.makedirs(name)
                       
if __name__ == '__main__':
    Create(NAME,NUMBER_LAYERS,NUMBER_PARAMS)
    print('save in database')
    print('copy Readme.md.example in folder '+str(NAME)+' and edit')
    

 
        