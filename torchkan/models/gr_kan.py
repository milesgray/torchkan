import torch
import torch.nn as nn

from .lkan import LKAN  # Assuming you have a PyTorch version of this

class AddAndNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_layer = nn.LayerNorm(None)  # size will be set when first called
    
    def forward(self, x, y):
        return self.norm_layer(x + y)

class Gate(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.dense_layer = nn.Linear(self.size, self.size)
        self.gated_layer = nn.Linear(self.size, self.size)

    def forward(self, x):
        dense_output = self.dense_layer(x)
        gated_output = torch.sigmoid(self.gated_layer(x))
        return dense_output * gated_output

class GRKAN(nn.Module):
    def __init__(self, input_size, output_size=None, activation=None, dropout=0.1, use_bias=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size if output_size is not None else input_size
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias

        self.skip_layer = nn.Linear(input_size, self.output_size)
        self.hidden_layer_1 = LKAN(self.output_size, self.output_size, 
                                   base_activation='elu', dropout=dropout, 
                                   use_bias=use_bias, use_layernorm=False)
        self.hidden_layer_2 = LKAN(self.output_size, self.output_size, 
                                   dropout=dropout, use_bias=use_bias, 
                                   use_layernorm=False)
        self.gate_layer = Gate(self.output_size)
        self.add_and_norm_layer = AddAndNorm()

    def forward(self, x):
        skip = self.skip_layer(x)
        
        hidden = self.hidden_layer_1(x)
        hidden = self.hidden_layer_2(hidden)
        hidden = self.dropout(hidden)
        gating_output = self.gate_layer(hidden)
        output = self.add_and_norm_layer(skip, gating_output)
        if self.activation is not None:
            output = self.activation(output)
        return output

class GRN(nn.Module):
    def __init__(self, hidden_layer_size, output_size=None, activation=None, dropout=0.1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size if output_size is not None else hidden_layer_size
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.skip_layer = nn.Linear(hidden_layer_size, self.output_size)
        self.hidden_layer_1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.hidden_layer_2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.gate_layer = Gate(self.output_size)
        self.add_and_norm_layer = AddAndNorm()

    def forward(self, x):
        skip = self.skip_layer(x)
        
        hidden = torch.nn.functional.elu(self.hidden_layer_1(x))
        hidden = self.hidden_layer_2(hidden)
        hidden = self.dropout(hidden)
        gating_output = self.gate_layer(hidden)
        output = self.add_and_norm_layer(skip, gating_output)
        if self.activation is not None:
            output = self.activation(output)
        return output