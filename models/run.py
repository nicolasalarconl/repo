from model import Model
from autoencoder import Autoencoder as net1
import sys
import cupy as cp
from matplotlib import pyplot as plt
sys.path.append("../database")
from database import DataBase

###############
size_figure = int(sys.argv[1])
str_type_psf = str(sys.argv[2]) # 'psf_real, 'psf_gauss'
start =  int(sys.argv[3])
stop = int(sys.argv[4])
count_model = int(sys.argv[5]) 
## GPU PARAMS ###
device_1  = 'cuda:0'
device_2  = 'cuda:1'
devices = [device_1,device_2]
#### DATABASE ######
path_db = '../database/database.db'
db = DataBase(path_db)
#db.db_reset()
##########DATASET ##### 
size_figure = size_figure
start = start
stop = stop
len_dataset = stop - start
## psf ####
type_psf = str_type_psf+'_'+str(size_figure)+'x'+str(size_figure)
########################

## NET PARAMS ########
perc_train = 0.7
perc_validation = 0.2  
perc_test = 0.1
batch_test = 1
batch_train = 10 
batch_validation = batch_train

def get_batch(batch_train,out_in,size_figure):
    if(size_figure == 640 and out_in == 640):
        batch_train = len(devices)*1
    elif (size_figure == 640 and out_in == 220):
        batch_train = len(devices)*2  
    elif(size_figure == 640 and out_in == 128):
        batch_train = len(devices)*6
    elif(size_figure == 640 and out_in == 64):
        batch_train = len(devices)*15 
    batch_validation = batch_train
    return batch_train,batch_validation

epochs = 10
learning_rate = 1e-3 #,1e-4,1e-5]
########################

for idx in cp.arange(0,count_model,1):
    try:
        print('model :'+str(idx)+'/'+str(count_model))
        #### MODELS #### 
            # Architecture
            # params 
            #ARCHITECTURE #####
        name_architecture = 'autoencoder1'
        id_architecture = 1
        db_architecture =(id_architecture,name_architecture)
        db.insert_architecture(db_architecture)
            #PARAMS #####
        id_param,k1,k2,k3,k4,k5,k6,k7,k8,p1,p2,p3,p4,s1,s2,s3,s4,out_in = db.get_params_random(size_figure)
        batch_train,batch_validation = get_batch(batch_train,out_in,size_figure)
        db_model = (id_architecture,id_param) #,size_figure,type_psf, epochs, learning_rate,batch_train,len_dataset)
        id_model = db.insert_model(db_model)
        net = net1(size = size_figure,out_in = out_in,
                       k1=k1,p1=p1,k2=k2,p2=p2,k3=k3,p3=p3,k4=k4,p4=p4,
                       k5=k5,s1=s1,k6=k6,s2=s2,k7=k7,s3=s3,k8=k8,s4=s4)     
        m = Model(
            size_figure = size_figure,
            type_psf = type_psf,
            num_epochs = epochs,
            learning_rate =  learning_rate , 
            start = start,
            stop = stop,
            devices = devices,
            perc_train =   perc_train,
            perc_validation =  perc_validation,
            perc_test  = perc_test,
            batch_train =  batch_train,
            batch_validation =  batch_validation,
            batch_test = batch_test,
            net = net
        )

        db_execution=(id_model,size_figure,type_psf,epochs,learning_rate ,start,stop,len(devices),perc_train,perc_validation,perc_test ,batch_train,batch_validation,batch_test,0,0,0,0,0,0 ,0,0)
        id_execution = db.insert_execution(db_execution)

        net,train_loss,valid_loss,time_train =m.run_train(net = net,start = 
                                                          start, stop = stop)

        db.insert_losses(id_execution,train_loss,'train')
        db.insert_losses(id_execution,valid_loss,'validation')

        db.update_train_time(id_execution,time_train)

        net,psnr_output,psnr_dirty,psnr_diff,time_test = m.run_test(net = net,start = start, stop = stop) 

        avg_psnr_dirty = cp.asnumpy(cp.average(psnr_dirty)).item()
        std_psnr_dirty = cp.asnumpy(cp.std(cp.array(psnr_dirty))).item()
        db.update_psnr_dirty(id_execution,avg_psnr_dirty,std_psnr_dirty)
        db.insert_psnr(id_execution,psnr_dirty,'dirty')

        avg_psnr_output = cp.asnumpy(cp.average(psnr_dirty)).item()
        std_psnr_output = cp.asnumpy(cp.std(cp.array(psnr_output))).item()
        db.update_psnr_output(id_execution,avg_psnr_output,std_psnr_output)
        db.insert_psnr(id_execution,psnr_output,'output')

        avg_psnr_diff = cp.asnumpy(cp.average(psnr_dirty)).item()
        std_psnr_diff  = cp.asnumpy(cp.std(cp.array(psnr_diff))).item()
        db.update_psnr_diff(id_execution,avg_psnr_diff,std_psnr_diff)
        db.insert_psnr(id_execution,psnr_diff,'diff')

        db.update_test_time(id_execution,time_test)
    except ValueError:
        print(ValueError)

