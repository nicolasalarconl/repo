import sys
sys.path.append("../../database/experiments_1")
from database import DataBase
from config import get_sizes,get_psfs,get_outs,get_kernel,get_stride,get_stride,get_padding,get_padding,get_path_db,get_architectures,get_models,get_executions,get_batchs,get_batch_test,get_epochs,get_learning_rates,get_start,get_sizes_dataset,get_devices,get_types_psf,get_path_images,get_perc_train,get_perc_validation,get_perc_test
import random
sys.path.append("../../dataset") 
from dataset import Dataset

sys.path.append("../../architectures/architecture_1") ## eliminar scripts ..
from autoencoder import Autoencoder as net1

sys.path.append("../../models") ## eliminar scripts ..
from model import Model 

import cupy as cp

#### VAR ####

## database ### 
PATH_DB=get_path_db()
DB =  DataBase(PATH_DB)
#DB.reset()
## Architectures ## 
ARCHITECTURES =  get_architectures()
##Params Model ## 
KERNEL = get_kernel()
STRIDE =  get_stride()
PADDING = get_padding()
OUTS = get_outs()
#Models#
MODELS = get_models()
SIZES = get_sizes()
PSFS =  get_psfs()
##PATH IMAGES #
PATH_IMAGES = get_path_images()
TYPES_PSF =  get_types_psf()
#HyperParams
EPOCHS = get_epochs()
BATCH_TEST = get_batch_test()
BATCHS = get_batchs()
LEARNING_RATES = get_learning_rates()
START = get_start()
SIZES_DATASET = get_sizes_dataset()
PERC_TRAIN =get_perc_train()
PERC_VALIDATION = get_perc_validation()
PERC_TEST =get_perc_test()
#Executions
EXECUTIONS = get_executions() 
## Devices ###
DEVICES = get_devices()
## TOTAL 
TOTAL=len(DEVICES)*len(ARCHITECTURES)*len(OUTS)*len(MODELS)*len(SIZES)*len(PSFS)*len(BATCHS)*len(LEARNING_RATES)*len(SIZES_DATASET)*len(EXECUTIONS)



def configure_parameters(out):
    k1 = random.choice(KERNEL)
    k2 = random.choice(KERNEL)
    k3 = random.choice(KERNEL)
    k4 = random.choice(KERNEL)
    k5 = random.choice(KERNEL)
    k6 = random.choice(KERNEL) 
    k7 = random.choice(KERNEL)
    k8 = random.choice(KERNEL)
    k9 = random.choice(KERNEL)
    s1 = random.choice(STRIDE)
    s2 = random.choice(STRIDE)
    s3 = random.choice(STRIDE)
    s4 = random.choice(STRIDE)
    p1 = random.choice(PADDING)
    p2 = random.choice(PADDING)
    p3 = random.choice(PADDING)
    p4 = random.choice(PADDING)
    p5 = random.choice(PADDING)

    values = (int(k1),int(k2),int(k3),int(k4),int(k5),int(k6),int(k7),int(k8),int(k9),int(p1),int(p2),int(p3),int(p4),int(p5),int(s1),int(s2),int(s3),int(s4),int(out))    
    values = (3,3,3,3,3,3,2,2,3,1,1,1,1,1,2,2,2,2,64)
    return values

def create_new_model(out):
    params = configure_parameters(out)
    exist = False #DB.exists_params(params)
    if (exist):
        return create_new_model(out)
    else:
        finish_model = 0
        id_params = DB.insert_params(params)
        id_model  = DB.insert_model((id_params,finish_model))
        return id_model,params

    
def insert_losses(id_execution,losses,type_loss):
    for loss in losses:
        db_loss = (id_execution,loss,type_loss)
        DB.insert_loss(db_loss)
def insert_psnrs(id_execution,psnrs,type_psnr):
    for psnr in psnrs:
        db_psnr = (id_execution,psnr,type_psnr)
        DB.insert_psnr(db_psnr)
        
        
def new_execution(id_execution,epoch,learning_rate,devices,train_data,validation_data,test_data,size_figure,params):
    
    net = net1(size_figure = size_figure,
               params = params)  
    m = Model(num_epochs = epoch,
              learning_rate = learning_rate , 
              devices = devices,
              net = net
             )
    net,train_loss,valid_loss,time_train,memory_train = m.train(net =net,
                                                                trainLoader= train_data,
                                                                validationloader=validation_data)
    
    
    insert_losses(id_execution,train_loss,'train')
    insert_losses(id_execution,valid_loss,'validation')
    
    
    
    net,psnr_output,psnr_dirty,psnr_diff,time_test,memory_test = m.test(net =net,
                                                                        testLoader= test_data)
    
    insert_psnrs(id_execution,psnr_output,'output')
    insert_psnrs(id_execution,psnr_dirty,'dirty')
    insert_psnrs(id_execution,psnr_diff,'diff')
    avg_output = cp.asnumpy(cp.average(psnr_output)).item()
    std_output = cp.asnumpy(cp.std(cp.array(psnr_output))).item()
    avg_dirty = cp.asnumpy(cp.average(psnr_dirty)).item()
    std_dirty = cp.asnumpy(cp.std(cp.array(psnr_dirty))).item()
    avg_diff = cp.asnumpy(cp.average(psnr_diff)).item()
    std_diff = cp.asnumpy(cp.std(cp.array(psnr_diff))).item()
    db_result = (id_execution,len(devices),time_train,time_test,
                 memory_train, memory_test,avg_output ,std_output,
                 avg_dirty,std_dirty,avg_diff,std_diff)
    
    id_result = DB.insert_result(db_result)
    return id_result
   
idx = 1

for D in DEVICES:
    for S in SIZES:
        for P in PSFS:
            for TP in TYPES_PSF:
                for SD in SIZES_DATASET:
                    for B in BATCHS:  
                        data = Dataset(devices = D,size_figure = S,path = PATH_IMAGES+str(S),
                                       psf =P,type_psf = TP,start= START,stop = SD,
                                       perc_train = PERC_TRAIN, batch_train =B,
                                       perc_validation = PERC_VALIDATION, batch_validation =B,
                                       perc_test = PERC_TEST,batch_test = BATCH_TEST)
                        train_data = data.train()
                        validation_data = data.validation()
                        test_data = data.test()
                        db_dataset = (START,SD,P,TP) 
                        id_dataset  = DB.insert_dataset(db_dataset)
                        for O in OUTS:
                            for M in MODELS:   
                                id_model,params = create_new_model(O)
                                for L in LEARNING_RATES:
                                    for EP in EPOCHS:
                                        db_hyperparameter = (EP,L,PERC_TRAIN,B,PERC_VALIDATION,B,PERC_TEST,BATCH_TEST)
                                        id_hyperparameter = DB.insert_hyperparameter(db_hyperparameter)
                                        for E in EXECUTIONS:
                                            finish_execution = 0
                                            db_execution = (id_model,id_hyperparameter,id_dataset,finish_execution)
                                            id_execution = DB.insert_execution(db_execution)
                                            id_result = new_execution(id_execution,EP,L,D,
                                                                      train_data,validation_data,
                                                                      test_data,S,params)
                                            finish_execution = 1
                                            DB.update_execution_finish(id_execution,finish_execution)
                                finish_model = 1
                                DB.update_model_finish(id_model,finish_model)