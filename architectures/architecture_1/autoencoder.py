# %%
import torch.nn as nn
import torch.nn.functional as F
import torch 
 
class Autoencoder(nn.Module):
    def __init__(self,size_figure,params):
        super(Autoencoder, self).__init__()
        self.size = size_figure
        self.k1 = params[0]
        self.k2 = params[1]
        self.k3 = params[2]
        self.k4 = params[3]
        self.k5 = params[4]
        self.k6 = params[5]
        self.k7 = params[6]
        self.k8 = params[7]
        self.k9 = params[8]
        self.p1 = params[9]
        self.p2 = params[10]
        self.p3 = params[11]
        self.p4 = params[12]
        self.p5 = params[13]
        self.s1 = params[14]
        self.s2 = params[15]
        self.s3 = params[16]
        self.s4 = params[17]
        self.out_in = params[18]        
        # encoder layers
        self.enc1 = nn.Conv2d(1, int(self.out_in), kernel_size=int(self.k1), padding=int(self.p1))
        self.enc2 = nn.Conv2d(int(self.out_in), int(self.out_in/2), kernel_size=int(self.k2), padding=int(self.p2))
        self.enc3 = nn.Conv2d(int(self.out_in/2), int(self.out_in/4),kernel_size=int(self.k3), padding=int(self.p3))
        self.enc4 = nn.Conv2d(int(self.out_in/4), int(self.out_in/8), kernel_size=int(self.k4), padding=int(self.p4))
        self.pool = nn.MaxPool2d(2, 2)
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(int(self.out_in/8),int(self.out_in/8), kernel_size=int(self.k5), stride=int(self.s1))
        self.dec2 = nn.ConvTranspose2d(int(self.out_in/8),int(self.out_in/4), kernel_size=int(self.k6), stride=int(self.s2))
        self.dec3 = nn.ConvTranspose2d(int(self.out_in/4),int(self.out_in/2), kernel_size=int(self.k7), stride=int(self.s3))
        self.dec4 = nn.ConvTranspose2d(int(self.out_in/2),int(self.out_in), kernel_size=int(self.k8), stride=int(self.s4))
        self.out = nn.Conv2d(int(self.out_in), int(1), kernel_size=int(self.k9), padding=int(self.p5))
        
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
        self.dec4.to(device)
        self.out.to(device)

        x = x.to(device)
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = self.pool(x) 
        #the latent space representation
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = torch.sigmoid(self.out(x))
        return x
