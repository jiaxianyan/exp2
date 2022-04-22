import torch
import torch.nn as nn
class GCN(nn.Module):
    def __init__(self,config):
        super(GCN, self).__init__()
        self.config = config

    def forward(self,data):

        return