import torch
import torch.nn as nn
import todevice as dv

class GeneratorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator=nn.Sequential(
            #shape=(128,1,1)
            nn.ConvTranspose2d(128,512,kernel_size=4,stride=1,padding=0,bias=False),
            #shape=(512,4,4)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #shape=(256,8,8)
            nn.ConvTranspose2d(256,128,kernel_size=4,padding=1,stride=2,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            #shape=(128,16,16)
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #shape=(64,32,32)
            nn.ConvTranspose2d(64,3,stride=2,kernel_size=4,padding=1,bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            #shape=(3,64,64)
            nn.Tanh()
        )
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # generator=dv.to_device(self.generator(x),dv.get_defaul_device())
        return self.generator(x)