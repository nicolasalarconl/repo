# %%
# %%
import cupy as cp

#### VAR ####
## database ## 
PATH_DB= '../../database/experiments_1/experiments_1.db'
# Architecture##
ARCHITECTURES = ['architecture_1']
#Images#
SIZES = [28] # ,640,1024]
PSFS = ['PSF_1']
PATH_IMAGES = '../../simulated_dataset/images/'
#Models#
OUTS =  [64] #,32]#,32,64,128]
KERNEL = cp.arange(1,3,1)
STRIDE = cp.arange(1,3,1)
PADDING = cp.arange(1,3,1)
MODELS = cp.arange(1,2,1)
#HyperParams
EPOCHS = [1]
BATCHS =  [2] #,4,8,16,32,64,128]
LEARNING_RATES = [1e-3]#,1e-4,1e-5]
SIZES_DATASET=  [5]#,16384,32768,65536,131072]
PERC_TRAIN = 0.7
PERC_VALIDATION = 0.2
PERC_TEST = 0.1
#Executions
EXECUTIONS = cp.arange(1,3,1)
## Devices ###
DEVICES = [ ['cuda:0']] #,['cuda:0','cuda:1']]

## architecture ## 
def get_path_db():
    return PATH_DB
def get_architectures():
    return ARCHITECTURES
## images ###
def get_sizes():
    return SIZES
def get_psfs():
    return PSFS
## MODELS ## 
def get_outs():
    return OUTS
def get_kernel():
    return KERNEL
def get_stride():
    return STRIDE
def get_padding():
    return PADDING
def get_models():
    return MODELS
def get_executions():
    return EXECUTIONS
def get_batchs():
    return BATCHS
def get_epochs():
    return EPOCHS
def get_learning_rates():
    return LEARNING_RATES
def get_sizes_dataset():
    return SIZES_DATASET
def get_devices():
    return DEVICES
def get_path_images():
    return PATH_IMAGES
def get_perc_train():
    return PERC_TRAIN
def get_perc_validation():
    return PERC_VALIDATION
def get_perc_test():
    return PERC_TEST
