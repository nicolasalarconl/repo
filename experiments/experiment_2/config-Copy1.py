# %%
# %%
import cupy as cp

#### VAR ####

# Architecture##
PATH_DB_ARCHITECTURES = ['../../database/architecture_1.db']
ARCHITECTURES = ['architecture_1']
#Images#
SIZES = [28,640,1024]
PSFS = ['PSF_1','PSF_2']

#Models#
OUTS = [64,128,256]
KERNEL = cp.arange(1,3,1)
STRIDE = cp.arange(1,3,1)
PADDING = cp.arange(1,3,1)
MODELS = 1000
#######



## architecture ## 
def get_path_db_architectures():
    return PATH_DB_ARCHITECTURES
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