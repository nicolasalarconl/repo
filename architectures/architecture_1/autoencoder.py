# %%
import torch.nn as nn
import torch.nn.functional as F
import torch 
 
class Autoencoder(nn.Module):
    def __init__(self,size,
                 out_in,
                 k1,p1,
                 k2,p2,
                 k3,p3,
                 k4,p4,
                 k5,s1,
                 k6,s2,
                 k7,s3,
                 k8,s4
                 ):
        super(Autoencoder, self).__init__()
        self.size = size
        self.out_in = int(out_in)
        # encoder layers
        self.enc1 = nn.Conv2d(1, int(self.out_in), kernel_size=int(k1), padding=int(p1))
        self.enc2 = nn.Conv2d(int(self.out_in), int(self.out_in/2), kernel_size=int(k2), padding=int(p2))
        self.enc3 = nn.Conv2d(int(self.out_in/2), int(out_in/4),kernel_size=int(k3), padding=int(p3))
        self.enc4 = nn.Conv2d(int(self.out_in/4), int(self.out_in/8), kernel_size=int(k4), padding=int(p4))
        self.pool = nn.MaxPool2d(2, 2)
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(int(self.out_in/8),int(self.out_in/8), kernel_size=int(k5), stride=int(s1))
        self.dec2 = nn.ConvTranspose2d(int(self.out_in/8),int(self.out_in/4), kernel_size=int(k6), stride=int(s2))
        self.dec3 = nn.ConvTranspose2d(int(self.out_in/4),int(self.out_in/2), kernel_size=int(k7), stride=int(s3))
        self.out = nn.Conv2d(out_in, 1, kernel_size=3, padding=1)
        self.initial = 0
        self.count_print = 0
    
  
    def set_padding_kernel(self,x,device):
        k  = 2
        s =  2
        p = int((((x.shape[2]-1)*s -self.size)+k)/2)
        if (p % 1 != 0):
            k = 3
            p =  int((((x.shape[2]-1)*s - self.size)+k)/2)
        self.dec4 = nn.ConvTranspose2d(int(self.out_in/2),  self.out_in, kernel_size=k, padding=p ,stride=s).to(device)
        

        
    def forward(self,x,device):
        self.enc1.to(device)
        self.enc2.to(device)
        self.enc3.to(device)
        self.enc4.to(device)
        self.pool.to(device)
        # decoder layers
        self.dec1.to(device)
        self.dec2.to(device)
        self.dec3.to(device)
        #self.dec4.to(device)
        self.out.to(device)
        x = x.to(device)
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = self.pool(x) # the latent space representation
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        self.set_padding_kernel(x,device)
        x = F.relu(self.dec4(x))
        x = torch.sigmoid(self.out(x))
        self.initial = self.initial + 1  
        return x
