# %%

from astropy.utils.data import download_file
from astropy.io import fits
import cupy as cp
import cv2


def psf_real(tamX,tamY):
            url = 'https://github.com/nicolasalarconl/InterferometryDeepLearning/blob/main/4_hd142_128x128_08.psf.fits?raw=true'
            image_link = download_file(url, cache=True )
            image = fits.getdata(image_link).astype(cp.float32)
            image = cp.reshape(image,[image.shape[2],image.shape[3]]) 
            image = cv2.resize(cp.asnumpy(image), dsize=(tamX, tamY), interpolation=cv2.INTER_CUBIC)
            image = cp.array(image)
            return image

def psf_gauss(tamX,tamY):
            x, y = cp.meshgrid(cp.linspace(-1,1,tamX), cp.linspace(-1,1,tamY))
            d = cp.sqrt(x*x+y*y)
            sigma, mu = 1/(tamX/2), 0.0
            gauss = cp.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
            return gauss
    