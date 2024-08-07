import torch
import torch.nn as nn
import torch.nn.functional as F

class GridInitializer:
    def __init__(self, grid_range, grid_size, spline_order):
        self.grid_range = grid_range
        self.grid_size = grid_size
        self.spline_order = spline_order

    def __call__(self, shape, dtype=None):
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        grid = torch.linspace(
            start=-self.spline_order * h + self.grid_range[0],
            end=(self.grid_size + self.spline_order) * h + self.grid_range[0],
            steps=self.grid_size + 2 * self.spline_order + 1
        )
        grid = grid.unsqueeze(0).unsqueeze(0).expand(1, shape[1], -1)
        return grid

class LKAN(nn.Module):
    """ Linear KAN
    """
    def __init__(
        self,
        units,
        grid_size=3,
        spline_order=3,
        base_activation='silu',
        grid_range=[-1, 1],
        dropout=0.,
        use_bias=True,
        use_layernorm=True,
    ):
        super().__init__()
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = getattr(F, base_activation)
        self.grid_range = grid_range
        self.use_bias = use_bias
        self.use_layernorm = use_layernorm
        
        if self.use_layernorm:
            self.layer_norm = nn.LayerNorm(None)  # None will be replaced in forward pass

        self.dropout = nn.Dropout(dropout)

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        self.other_dims = input_shape[1:-1]

        self.grid = nn.Parameter(
            GridInitializer(self.grid_range, self.grid_size, self.spline_order)(
                [1, self.in_features, self.grid_size + 2 * self.spline_order + 1]
            ),
            requires_grad=False
        )

        self.base_weight = nn.Parameter(torch.Tensor(self.units, self.in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        if self.use_bias:
            self.base_bias = nn.Parameter(torch.zeros(self.units))

        self.spline_weight = nn.Parameter(torch.Tensor(self.units, self.in_features * (self.grid_size + self.spline_order)))
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

    def forward(self, x):
        if not hasattr(self, 'base_weight'):
            self.build(x.size())
        
        if self.use_layernorm:
            if self.layer_norm.normalized_shape != x.size()[1:]:
                self.layer_norm = nn.LayerNorm(x.size()[1:]).to(x.device)
            x = self.layer_norm(x)
        
        base_output = F.linear(self.base_activation(x), self.base_weight.t(), self.base_bias if self.use_bias else None)
        spline_output = torch.matmul(self.b_splines(x), self.spline_weight.t())
        
        return self.dropout(base_output) + self.dropout(spline_output)

    def b_splines(self, x):
        batch_size = x.size(0)
        x_expanded = x.unsqueeze(-1)
        
        grid_expanded = self.grid.expand(batch_size, self.in_features, -1)
        for dim in reversed(self.other_dims):
            grid_expanded = grid_expanded.unsqueeze(1).expand(-1, dim, -1, -1)

        bases = ((x_expanded >= grid_expanded[..., :-1]) & (x_expanded < grid_expanded[..., 1:])).float()
        
        for k in range(1, self.spline_order + 1):
            left_denominator = grid_expanded[..., k:-1] - grid_expanded[..., :-(k + 1)]
            right_denominator = grid_expanded[..., k + 1:] - grid_expanded[..., 1:-k]
            
            left = (x_expanded - grid_expanded[..., :-(k + 1)]) / left_denominator
            right = (grid_expanded[..., k + 1:] - x_expanded) / right_denominator
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        
        bases = bases.reshape(batch_size, *bases.shape[1:-2], -1)
        return bases