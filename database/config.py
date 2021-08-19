from database import DataBase
import sys

PATH_DB = 'database.db'
ACTION =  str(sys.argv[1]) 
DB = DataBase(PATH_DB)
                   
if __name__ == '__main__':
    if (ACTION == 'reset'):
        DB.reset()
        print('DB: reset all')
    if (ACTION == 'create'):
        DB.create_all_tables()
        print('DB: create all tables')

    if (ACTION == 'delete'):
        DB.delete_all_tables()
        print('DB: delete all tables')
        
      