import sys
sys.path.append("../../database")
from database import DataBase
from config import get_all_size,get_all_psf,get_outs
## DATABASE ### 
PATH_DB = '../../database/architecture_1.db'
DB = DataBase(PATH_DB)
######

##Params Model ## 
MIN_KERNEL = 1
MAX_KERNEL = 6
MIN_STRIDE = 
MAX_STRIDE =
MIN_PADDING = 
MAX_PADDING =
OUTS = 
#####
### MODELS  ####
MODELS = 2
SIZES = get_all_size()
DIRTYS =  get_all_dirtys()

def create_new_model():
    random_kernel  = cp.arange(MIN_KERNEL,MAX_KERNEL,1)
    random_stride  = cp.arange(MIN_STRIDE,MAX_STRIDE,1)
    random_stride  = cp.arange(MIN_PADDING,MAX_PADDING,1)
    randoms_out  = [64,128,256]
    k1 = random.choice(random_kernel)
    k2 = random.choice(random_kernel)
    k3 = random.choice(random_kernel)
    k4 = random.choice(random_kernel)
    k5 = random.choice(random_kernel)
    k6 = random.choice(random_kernel) 
    k7 = random.choice(random_kernel)
    k8 = random.choice(random_kernel)
    s1 = random.choice(randoms)
    s2 = random.choice(randoms)
    s3 = random.choice(randoms)
    s4 = random.choice(randoms)
    p1 = random.choice(randoms)
    p2 = random.choice(randoms)
    p3 = random.choice(randoms)
    p4 = random.choice(randoms)
    out_in = random.choice(randoms_out)
    values = (int(k1),int(k2),int(k3),int(k4),int(k5),int(k6),int(k7),int(k8),int(p1),int(p2),int(p3),int(p4),int(s1),int(s2),int(s3),int(s4),int(out_in))
    if (DB.exists_params(values)):
        create_new_model()
    else:
        id_params = DB.insert_params(values)
        return id_params
    
   
cursorObj.execute('SELECT * FROM params_model WHERE k1 = ? AND k2 = ? AND k3 = ? AND k4 = ? AND k5 = ? AND k6 = ? AND k7 = ? AND k8 = ? AND p1 = ? AND p2 = ? AND p3 = ? AND p4 = ? AND s1 = ? AND s2 = ? AND s3 = ? AND s4 = ? AND out_in = ? ',values) 
        rows = cursorObj.fetchall()
        if (len(rows) == 0):
            id_param = self.insert_params(values)
            cursorObj.execute('SELECT * FROM params_model where id ='+str(id_param))
            return cursorObj.fetchall()[0]
        else: 
            return self.get_params_random(size_figure)    
    
ARCHITECTURE =  DB.get_all_architectures()
for A in ARCHITECTURE:
    for M in MODELS:
        id_params = create_new_model()
        DB.insert()
        for S in SIZES:
            for D in DIRTYS:
                execution(id_params)
        
        
            

    
 

