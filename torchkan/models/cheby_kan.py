import torch
import torch.nn as nn

from ..layers import ChebyKANLayer
    
class ChebyKAN(nn.Module):
    def __init__(
            self, 
            input_size, 
            output_size
    ):
        super().__init__()
        self.input_size = input_size
        self.kans = [
            ChebyKANLayer(input_size, 32, 4, norm=True),
            ChebyKANLayer(32, 16, 4, norm=True),
            ChebyKANLayer(16, output_size, 4)
        ]

    def forward(self, x):
        x = x.view(-1, self.input_size)  # Flatten the images
        x = self.kans(x)[0]
        x = self.kans(x)[1]
        x = self.kans(x)[2]
        return x