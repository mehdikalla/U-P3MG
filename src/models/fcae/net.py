import torch as tc
import torch.nn as nn

from src.models.FC_block import FC_block

class FCAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = FC_block([100,50,25,12,12,25,50,100])
    
    def forward(self, x):
        return self.layers(x)
    
