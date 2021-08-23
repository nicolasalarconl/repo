import sys
sys.path.append("../../database/experiments_1")
from database import DataBase
from config import get_sizes,get_psfs,get_outs,get_kernel,get_stride,get_stride,get_padding,get_padding,get_path_db,get_architectures,get_models,get_executions,get_batchs,get_epochs,get_learning_rates,get_sizes_dataset,get_devices,get_path_images,get_perc_train,get_perc_validation,get_perc_test
import random
sys.path.append("../../dataset/scripts") ## eliminar scripts ..
from dataset import Dataset

sys.path.append("../../architectures/architecture_1") ## eliminar scripts ..
from autoencoder import Autoencoder as net1

sys.path.append("../../models") ## eliminar scripts ..
from model import Model 

#### VAR ####

## database ### 
PATH_DB=get_path_db()
DB =  DataBase(PATH_DB)
DB.reset()
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
#HyperParams
EPOCHS = get_epochs()
BATCHS = get_batchs()
LEARNING_RATES = get_learning_rates()
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
        id_params = DB.insert_params(params)
        id_model  = DB.insert_model((id_params,0))
        return id_model,params

def new_execution(epoch,learning_rate,devices,train_data,validation_data,test_data,size_figure,params):
    net = net1(size_figure = size_figure,params = params)  
    m = Model(num_epochs = epoch,
              learning_rate = learning_rate , 
              devices = devices,
              net = net
             )
    net,train_loss,valid_loss,time_train,memory_train = m.train(net =net,
                                                                trainLoader= train_data,validationloader=validation_data)
    net,psnr_output,psnr_dirty,psnr_diff,time_test,memory_test = m.test(net =net,testLoader= test_data)
   
   
idx = 1
for D in DEVICES:
    for S in SIZES:
        for P in PSFS:
            for SD in SIZES_DATASET:
                for B in BATCHS:  
                    data = Dataset(devices = D,size_figure = S,path = PATH_IMAGES+str(S),
                                   psf =P,type_psf = 'Real',start= 0,stop = SD,
                                   perc_train = PERC_TRAIN, batch_train =B,
                                   perc_validation = PERC_VALIDATION, batch_validation =B,
                                   perc_test = PERC_TEST,batch_test = 1)
                    train_data = data.train()
                    validation_data = data.validation()
                    test_data = data.test()
                    for O in OUTS:
                        for M in MODELS:   
                            id_model,params = create_new_model(O)
                            for L in LEARNING_RATES:
                                for EP in EPOCHS:
                                    for E in EXECUTIONS:
                                        new_execution(EP,L,D,train_data,validation_data,test_data,S,params)