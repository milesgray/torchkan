import torch
import torch.nn as nn

import iisignature  # Assuming there's a PyTorch version of iisignature

from sigkan import LKAN, GRKAN, GRN  # Assuming these have been converted to PyTorch as well
from layers.sig import SigLayer

class SigKAN(nn.Module):
    def __init__(self, unit, sig_level, dropout=0.):
        super().__init__()
        self.unit = unit
        self.sig_level = sig_level
        self.sig_layer = SigLayer(self.sig_level)
        self.kan_layer = LKAN(unit, dropout=dropout, use_bias=False, use_layernorm=False)
        self.sig_to_weight = GRKAN(unit, activation='softmax', dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def build(self, input_shape):
        _, seq_length, _ = input_shape
        self.time_weighting_kernel = nn.Parameter(torch.randn(seq_length, 1))

    def forward(self, x):
        if not hasattr(self, 'time_weighting_kernel'):
            self.build(x.shape)
        
        x = self.time_weighting_kernel * x
        sig = self.sig_layer(x)
        weights = self.sig_to_weight(sig)
        kan_out = self.kan_layer(x)
        kan_out = self.dropout(kan_out)
        return kan_out * weights.unsqueeze(2)

class SigDense(nn.Module):
    def __init__(self, unit, sig_level, dropout=0.):
        super().__init__()
        self.unit = unit
        self.sig_level = sig_level
        self.sig_layer = SigLayer(self.sig_level)
        self.dense_layer = nn.Linear(unit, unit)
        self.sig_to_weight = GRN(unit, activation='softmax', dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def build(self, input_shape):
        _, seq_length, _ = input_shape
        self.time_weighting_kernel = nn.Parameter(torch.randn(seq_length, 1))

    def forward(self, x):
        if not hasattr(self, 'time_weighting_kernel'):
            self.build(x.shape)
        
        x = self.time_weighting_kernel * x
        sig = self.sig_layer(x)
        weights = self.sig_to_weight(sig)
        dense_out = self.dense_layer(x)
        dense_out = self.dropout(dense_out)
        return dense_out * weights.unsqueeze(2)