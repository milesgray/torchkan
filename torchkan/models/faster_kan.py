""" https://github.com/hoangthangta/BSRBF_KAN/blob/main/models/faster_kan.py """
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import *
from torch.autograd import Function

class RSWAFFunction(Function):
    @staticmethod
    def forward(ctx, input, grid, inv_denominator, train_grid, train_inv_denominator):
        # Compute the forward pass
        diff = (input[..., None] - grid)
        diff_mul = diff.mul(inv_denominator)
        tanh_diff = torch.tanh(diff)
        tanh_diff_deriviative = -tanh_diff.mul(tanh_diff) + 1  # sech^2(x) = 1 - tanh^2(x)
        
        # Save tensors for backward pass
        ctx.save_for_backward(input, tanh_diff, tanh_diff_deriviative, diff, inv_denominator)
        ctx.train_grid = train_grid
        ctx.train_inv_denominator = train_inv_denominator
        
        return tanh_diff_deriviative

##### SOS NOT SURE HOW grad_inv_denominator, grad_grid ARE CALCULATED CORRECTLY YET
##### MUST CHECK https://github.com/pytorch/pytorch/issues/74802
##### MUST CHECK https://www.changjiangcai.com/studynotes/2020-10-18-Custom-Function-Extending-PyTorch/
##### MUST CHECK https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
##### MUST CHECK https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
##### MUST CHECK https://gist.github.com/Hanrui-Wang/bf225dc0ccb91cdce160539c0acc853a

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        _, tanh_diff, tanh_diff_deriviative, diff, inv_denominator = ctx.saved_tensors
        grad_grid = None
        grad_inv_denominator = None
        
        # Compute the backward pass for the input
        grad_input = -2 * tanh_diff * tanh_diff_deriviative * grad_output
        grad_input = grad_input.sum(dim=-1).mul(inv_denominator)
        
        # Compute the backward pass for grid
        if ctx.train_grid:
            grad_grid = -inv_denominator * grad_output.sum(dim=0).sum(dim=0)#-(inv_denominator * grad_output * tanh_diff_deriviative).sum(dim=0) #-inv_denominator * grad_output.sum(dim=0).sum(dim=0)

        # Compute the backward pass for inv_denominator        
        if ctx.train_inv_denominator:
            grad_inv_denominator = (grad_output* diff).sum() #(grad_output * diff * tanh_diff_deriviative).sum() #(grad_output* diff).sum() 

        return grad_input, grad_grid, grad_inv_denominator, None, None # same number as tensors or parameters



class ReflectionalSwitchFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -1.2,
            grid_max: float = 0.2,
            num_grids: int = 8,
            inv_denominator: float = 0.5,
            train_grid: bool = False,        
            train_inv_denominator: bool = False,
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.train_grid = torch.tensor(train_grid, dtype=torch.bool)
        self.train_inv_denominator = torch.tensor(train_inv_denominator, dtype=torch.bool) 
        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator, dtype=torch.float32), requires_grad=train_inv_denominator)  # Cache the inverse of the denominator

    def forward(self, x):
        return RSWAFFunction.apply(x, self.grid, self.inv_denominator, self.train_grid, self.train_inv_denominator)


class SplineLinear(nn.Linear):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            init_scale: float = 0.1, 
            **kwargs
    ) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kwargs)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)  # Using Xavier Uniform initialization


class FasterKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -1.2,
        grid_max: float = 0.2,
        num_grids: int = 8,
        exponent: int = 2,
        inv_denominator: float = 0.5,
        train_grid: bool = False,        
        train_inv_denominator: bool = False,
        spline_weight_init_scale: float = 0.667,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = ReflectionalSwitchFunction(grid_min, grid_max, num_grids, exponent, inv_denominator, train_grid, train_inv_denominator)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)

    def forward(self, x):
        x = self.layernorm(x)
        spline_basis = self.rbf(x).view(x.shape[0], -1)
        
        ret = self.spline_linear(spline_basis)
        
        return ret

class FasterKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -1.2,
        grid_max: float = 0.2,
        num_grids: int = 8,
        exponent: int = 2,
        inv_denominator: float = 0.5,
        train_grid: bool = False,        
        train_inv_denominator: bool = False,
        base_activation = None,
        spline_weight_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FasterKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                exponent = exponent,
                inv_denominator = inv_denominator,
                train_grid = train_grid ,
                train_inv_denominator = train_inv_denominator,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class BasicResBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            stride: int = 1
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)

        return out

class SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(
            self,
            in_channels: int, 
            out_channels: int, 
            kernel_size: int,
            stride: int = 1,
            padding: int = 0
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class EnhancedFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.initial_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Increased number of filters
            nn.ReLU(),
            nn.BatchNorm2d(32),  # Added Batch Normalization
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),  # Added Dropout
            BasicResBlock(32, 64),
            SEBlock(64, reduction=16),  # Squeeze-and-Excitation block
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),  # Added Dropout
            DepthwiseSeparableConv(64, 128, kernel_size=3),  # Increased number of filters
            nn.ReLU(),
            BasicResBlock(128, 256),
            SEBlock(256, reduction=16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),  # Added Dropout
            SelfAttention(256),  # Added Self-Attention layer
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for fully connected layers
        return x

class FasterKANvolver(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -1.2,
        grid_max: float = 0.2,
        num_grids: int = 8,
        exponent: int = 2,
        inv_denominator: float = 0.5,
        train_grid: bool = False,        
        train_inv_denominator: bool = False,
        base_activation = None,
        spline_weight_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        
        # Feature extractor with Convolutional layers
        self.feature_extractor = EnhancedFeatureExtractor()
        """
        nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1 input channel (grayscale), 16 output channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        """

        # Calculate the flattened feature size after convolutional layers
        flat_features = 256 # XX channels, image size reduced to YxY
        
        # Update layers_hidden with the correct input size from conv layers
        layers_hidden = [flat_features] + layers_hidden

        # Define the FasterKAN layers
        self.faster_kan_layers = nn.ModuleList([
            FasterKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                exponent=exponent,
                inv_denominator = 0.5,
                train_grid = False,        
                train_inv_denominator = False,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])   

    def forward(self, x):
        # Reshape input from [batch_size, 784] to [batch_size, 1, 28, 28] for MNIST [batch_size, 1, 32, 32] for C
        x = x.view(-1, 3, 32,32)
        # Apply convolutional layers
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the output from the conv layers
        
        # Pass through FasterKAN layers
        for layer in self.faster_kan_layers:
            x = layer(x)
        
        return x