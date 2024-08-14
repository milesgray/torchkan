""" Various KAN layers"""
import torch
import torch.nn as nn
import numpy as np

# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    """ https://github.com/SpaceLearner/JacobiKAN/blob/main/ChebyKANLayer.py """
    def __init__(self, input_dim, output_dim, degree, norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        self.norm = nn.LayerNorm(self.output_dim) if norm else norm

        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.input_dim))  # shape = (batch_size, input_dim)
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Chebyshev polynomial tensors
        cheby = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
        if self.degree > 0:
            cheby[:, :, 1] = x
        for i in range(2, self.degree + 1):
            cheby[:, :, i] = 2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()
        # Compute the Chebyshev interpolation
        y = torch.einsum('bid,iod->bo', cheby, self.cheby_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.output_dim)
        if self.norm:
            y = self.norm(y)
        return y
    
# This is inspired by Kolmogorov-Arnold Networks but using Jacobian polynomials instead of splines coefficients
class JacobiKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0, norm=False):
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        self.norm = nn.LayerNorm(self.output_dim) if norm else norm

        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Since Jacobian polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0: ## degree = 0: jacobi[:, :, 0] = 1 (already initialized) ; degree = 1: jacobi[:, :, 1] = x ; d
            jacobi[:, :, 1] = ((self.a-self.b) + (self.a+self.b+2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k  = (2*i+self.a+self.b)*(2*i+self.a+self.b-1) / (2*i*(i+self.a+self.b))
            theta_k1 = (2*i+self.a+self.b-1)*(self.a*self.a-self.b*self.b) / (2*i*(i+self.a+self.b)*(2*i+self.a+self.b-2))
            theta_k2 = (i+self.a-1)*(i+self.b-1)*(2*i+self.a+self.b) / (i*(i+self.a+self.b)*(2*i+self.a+self.b-2))
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[:, :, i - 1].clone() - theta_k2 * jacobi[:, :, i - 2].clone()  # 2 * x * jacobi[:, :, i - 1].clone() - jacobi[:, :, i - 2].clone()
        # Compute the Jacobian interpolation
        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        # layerNorm
        if self.norm:
            y = self.norm(y)
        return y
    

class ReLUKANLayer(nn.Module):
    """ https://github.com/quiqi/relu_kan/blob/main/torch_relu_kan.py """
    def __init__(self, 
                 input_size: int, 
                 g: int, # phase_num
                 k: int, # step
                 output_size: int, 
                 train_ab: bool = True):
        super().__init__()
        self.input_size, self.output_size = input_size, output_size
        self.g, self.k  = g, k
        
        self.r = 4*g*g / ((k+1)*(k+1))        
        phase_low = np.arange(-k, g) / g
        phase_high = phase_low + (k+1) / g
        self.phase_low = nn.Parameter(
            torch.Tensor(
                np.array(
                    [phase_low for i in range(input_size)]
                )
            ), requires_grad=train_ab)
        self.phase_high = nn.Parameter(
            torch.Tensor(
                np.array(
                    [phase_high for i in range(input_size)]
                )
            ), requires_grad=train_ab)
        self.conv = nn.Conv2d(1, output_size, (g+k, input_size))

    def forward(self, x):
        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_high - x)
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.conv(x)
        x = x.reshape((len(x), self.output_size, 1))
        return x
    
    def show_base(self, x_dim=600, y_dim=1024):     
        import matplotlib.pyplot as plt   
        x = torch.Tensor([np.arange(-x_dim, y_dim+x_dim) / y_dim]).T
        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_high - x)
        y = x1 * x1 * x2 * x2 * self.r * self.r
        for i in range(self.g+self.k):
            plt.plot(x, y[:, i:i+1].detach(), color='black')
        plt.show()