import torch.nn as nn 
import torch
import todevice as dv
#new shape= (w -ker +2*pad)/stride +1
class DiscriminatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.discriminator=nn.Sequential(
            #shape=(3,64,64)
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            #shape=(64,32,32)
            nn.LeakyReLU(0.2,inplace=True),
            #shape=(64,32,32)
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias=False),
            #shape=(128,16,16)  
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128,256,kernel_size=4,padding=1,stride=2,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            #shape=(256,8,8)
            nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(512,1,stride=1,kernel_size=4,padding=0,bias=False),
            #shape=(1,1,1)
            nn.Flatten(),
            nn.Sigmoid()
        )
    def forward(self,x:torch.Tensor)-> torch.Tensor:
        # discriminator=dv.to_device(self.discriminator,dv.get_defaul_device())
        return self.discriminator(x)