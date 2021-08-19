from dataset import Dataset
import sys
device = int(sys.argv[1])
size_figure = int(sys.argv[2]) # prints python_script.py
start = int(sys.argv[3]) # prints var1
stop   = int(sys.argv[4]) # prints var2
sys.path.append("../")
type_psf_gauss = 'psf_gauss_'+str(size_figure)+'x'+str(size_figure)
type_psf_real = 'psf_real_'+str(size_figure)+'x'+str(size_figure)
data = Dataset(size_figure = size_figure,
            device = device)
data.create(start,stop)
