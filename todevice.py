# Copyright (c) 2026-present, Mohamed Chtourou.
# All rights reserved.
# This module defines utility functions for handling device placement of tensors and data loaders in PyTorch. It includes a function to determine the default device (CPU or GPU), a function to move data to the specified device, and a custom DataLoader class that automatically moves batches of data to the desired device during iteration. These utilities are used to facilitate training and inference of the GAN models implemented in the generative_model.py module.

import torch


def get_defaul_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)


class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl=dl
        self.device=device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b,self.device)
    def __len__(self):
        return len(self.dl)
