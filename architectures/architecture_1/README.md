# AUTOENCODER 



### Params
 * out_in,
  *k1,p1,
                 k2,p2,
                 k3,p3,
                 k4,p4,
                 k5,s1,
                 k6,s2,
                 k7,s3,
                 k8,s4
 * Numbero de capas
  layer
  numero de parametros ...
  imagen de la architectura
  
  
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
