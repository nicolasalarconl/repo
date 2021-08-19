# %%
#import cupy as cp
import sys
sys.path.append("../../scripts")
from auxiliar.save import save_psf

import cupy as cp
import cv2

from astropy.io import fits
from astropy.utils.data import download_file

DEVICE = int(sys.argv[1])
NAME = str(sys.argv[2]) 
SIZE_FIGURE = int(sys.argv[3]) 
URL = str(sys.argv[4]) 


class Create:
    #TODO : path save a path y size_image size figure
    def __init__(self,device,name_psf,size_figure):
        self.size_figure = size_figure
        self.device = self.init_device(device)
        self.path_gauss = name_psf+'/PSF_Gauss'
        self.path_real = name_psf+'/PSF_Real'
        
    def init_device(self,device):
        cp.cuda.Device(device).use()
        return device
    
    def psf_real(self,url):
        image_link = download_file(url, cache=True )
        image = fits.getdata(image_link).astype(cp.float32)
        image = cp.reshape(image,[image.shape[2],image.shape[3]]) 
        image = cv2.resize(cp.asnumpy(image), dsize=(self.size_figure, self.size_figure), interpolation=cv2.INTER_CUBIC)
        psf = cp.array(image)
        type_psf = 'psf_real'
        save_psf(self.path_real,type_psf,cp.asnumpy(psf))
        return psf
     
    def radius(self,psf):
        return 1
        
    def psf_gauss(self,radius): 
        x, y = cp.meshgrid(cp.linspace(-1,1,self.size_figure), cp.linspace(-1,1,self.size_figure))
        d = cp.sqrt(x*x+y*y)
        sigma, mu = 1/(self.size_figure/round(self.size_figure*0.1)), 0.0
        gauss = cp.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        psf = cp.asnumpy(gauss)
        type_psf = 'psf_gauss'
        save_psf(self.path_gauss,type_psf,psf)
    
if __name__ == '__main__':
    print('creating PSF....')
    cr = Create(DEVICE,NAME,SIZE_FIGURE)
    psf_real = cr.psf_real(URL)
    radius = cr.radius(psf_real)
    psf_gauss =  cr.psf_gauss(radius)
    print('Finished')


    