""" https://github.com/quiqi/relu_kan/blob/main/torch_relu_kan.py """
import numpy as np
import torch
import torch.nn as nn

from torchkan.layers import ReLUKANLayer


class ReLUKAN(nn.Module):
    def __init__(self, width, grid, k):
        super().__init__()
        self.width = width
        self.grid = grid
        self.k = k
        self.rk_layers = nn.ModuleList([
            ReLUKANLayer(width[i], grid, k, width[i+1]) 
            for i in range(len(width) - 1)
        ])

    def forward(self, x):
        for rk_layer in self.rk_layers:
            x = rk_layer(x)
        return x

    def show_base(self, idx, x_dim=600, y_dim=1024):        
        self.rk_layers[idx].show_base(x_dim, y_dim)