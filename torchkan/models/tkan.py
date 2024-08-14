import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union

class KANLinear(nn.Module):
    def __init__(self, output_dim, spline_order=None, use_layernorm=True):
        super().__init__()
        self.output_dim = output_dim
        self.spline_order = spline_order
        self.use_layernorm = use_layernorm
        
        # Initialize layers here
        # This is a placeholder and should be implemented based on the original KANLinear
        self.linear = nn.Linear(output_dim, output_dim)
        if use_layernorm:
            self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        x = self.linear(x)
        if self.use_layernorm:
            x = self.layer_norm(x)
        return x

class TKANCell(nn.Module):
    """Cell class for the TKAN layer.
    Modification of the LSTM implementation in keras 3 in order to provide fully seamless integration with TF, torch and jax backend

    This class processes one step within the whole time sequence input, whereas
    `TKAN` processes the whole sequence.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
            Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
            applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation
            of the recurrent state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        unit_forget_bias: Boolean (default `True`). If `True`,
            add 1 to the bias of the forget gate at initialization.
            Setting it to `True` will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](
            https://github.com/mlresearch/v37/blob/gh-pages/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        seed: Random seed for dropout.

    Call arguments:
        inputs: A 2D tensor, with shape `(batch, features)`.
        states: A 2D tensor with shape `(batch, units)`, which is the state
            from the previous time step.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.

    Example:

    >>> inputs = np.random.random((32, 10, 8))
    >>> rnn = keras.layers.RNN(keras.layers.TKANCell(4))
    >>> output = rnn(inputs)
    >>> output.shape
    (32, 4)
    >>> rnn = keras.layers.RNN(
    ...    keras.layers.TKANCell(4),
    ...    return_sequences=True,
    ...    return_state=True)
    >>> whole_sequence_output, final_state = rnn(inputs)
    >>> whole_sequence_output.shape
    (32, 10, 4)
    >>> final_state.shape
    (32, 4)
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sub_kan_configs: Optional[List[Union[int, float, dict, str]]] = None,
        sub_kan_output_dim: Optional[int] = None,
        sub_kan_input_dim: Optional[int] = None,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        unit_forget_bias: bool = True,
        kernel_regularizer: Optional[str] = None,
        recurrent_regularizer: Optional[str] = None,
        bias_regularizer: Optional[str] = None,
        kernel_constraint: Optional[str] = None,
        recurrent_constraint: Optional[str] = None,
        bias_constraint: Optional[str] = None,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        seed: int = None,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sub_kan_configs = sub_kan_configs or [None]
        self.sub_kan_output_dim = sub_kan_output_dim or input_size
        self.sub_kan_input_dim = sub_kan_input_dim or input_size
        self.use_bias = use_bias
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.seed = seed if seed else 1337

        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.unit_forget_bias = unit_forget_bias
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.recurrent_constraint = recurrent_constraint
        self.bias_constraint = bias_constraint

        self.activation = getattr(torch, activation)
        self.recurrent_activation = getattr(torch, recurrent_activation)

        self.W_ih = nn.Parameter(torch.Tensor(input_size, hidden_size * 3))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 3))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size * 3))
        else:
            self.register_parameter('bias', None)

        self.tkan_sub_layers = nn.ModuleList()
        for config in self.sub_kan_configs:
            if config is None: 
                layer = KANLinear(self.sub_kan_output_dim, use_layernorm=True)
            elif isinstance(config, (int, float)):
                layer = KANLinear(self.sub_kan_output_dim, spline_order=config, use_layernorm=True)
            elif isinstance(config, dict):
                layer = KANLinear(self.sub_kan_output_dim, **config, use_layernorm=True)
            else:
                layer = nn.Linear(self.sub_kan_input_dim, self.sub_kan_output_dim)
            self.tkan_sub_layers.append(layer)

        self.sub_tkan_kernel = nn.Parameter(torch.Tensor(len(self.tkan_sub_layers), self.sub_kan_output_dim * 2))
        self.sub_tkan_recurrent_kernel_inputs = nn.Parameter(torch.Tensor(len(self.tkan_sub_layers), input_size, self.sub_kan_input_dim))
        self.sub_tkan_recurrent_kernel_states = nn.Parameter(torch.Tensor(len(self.tkan_sub_layers), self.sub_kan_output_dim, self.sub_kan_input_dim))
        
        self.aggregated_weight = nn.Parameter(torch.Tensor(len(self.tkan_sub_layers) * self.sub_kan_output_dim, hidden_size))
        self.aggregated_bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.orthogonal_(self.W_hh)
        if self.use_bias:
            nn.init.zeros_(self.bias)
            if self.unit_forget_bias:
                nn.init.ones_(self.bias[self.hidden_size:self.hidden_size*2])  # forget gate bias init to 1
        
        nn.init.orthogonal_(self.sub_tkan_kernel)
        nn.init.orthogonal_(self.sub_tkan_recurrent_kernel_inputs)
        nn.init.orthogonal_(self.sub_tkan_recurrent_kernel_states)
        nn.init.xavier_uniform_(self.aggregated_weight)
        nn.init.zeros_(self.aggregated_bias)

    def _generate_dropout_mask(self, inputs):
        if 0 < self.dropout < 1:
            torch.manual_seed(self.seed)
            return F.dropout(
                torch.ones_like(inputs),
                self.dropout,
            )
        return None

    def _generate_recurrent_dropout_mask(self, states):
        if 0 < self.recurrent_dropout < 1:
            torch.manual_seed(self.seed)
            return F.dropout(
                torch.ones_like(states),
                self.recurrent_dropout,
            )
        return None

    def forward(self, x, states):
        h_tm1 = states[0]  # Previous memory state
        c_tm1 = states[1]  # Previous carry state
        sub_states = states[2:]  # Previous states for sub-layers

        if self.training:
            self.seed = (self.seed + 1) % (2**32 - 1) 
            dp_mask = self._generate_dropout_mask(x)
            rec_dp_mask = self._generate_recurrent_dropout_mask(h_tm1)
            if dp_mask is not None:
                x *= dp_mask
            if rec_dp_mask is not None:
                h_tm1 *= rec_dp_mask

        if self.use_bias:
            gates = torch.matmul(x, self.W_ih) + \
                torch.matmul(h_tm1, self.W_hh) + self.bias
        else:
            gates = torch.matmul(x, self.W_ih) + \
                torch.matmul(h_tm1, self.W_hh)
        
        i, f, c = self.recurrent_activation(gates).chunk(3, 1)

        c = f * c_tm1 + i * self.activation(c)

        sub_outputs = []
        new_sub_states = []

        for idx, (sub_layer, sub_state) in enumerate(zip(self.tkan_sub_layers, sub_states)):
            sub_kernel_x = self.sub_tkan_recurrent_kernel_inputs[idx]
            sub_kernel_h = self.sub_tkan_recurrent_kernel_states[idx]
            agg_input = torch.matmul(x, sub_kernel_x) + \
                        torch.matmul(sub_state, sub_kernel_h)
            sub_output = sub_layer(agg_input)
            sub_recurrent_kernel_h, sub_recurrent_kernel_x = self.sub_tkan_kernel[idx].chunk(2, 0)
            new_sub_state = sub_recurrent_kernel_h * sub_output + \
                            sub_state * sub_recurrent_kernel_x

            sub_outputs.append(sub_output)
            new_sub_states.append(new_sub_state)

        aggregated_sub_output = torch.cat(sub_outputs, dim=-1)
        aggregated_input = F.linear(aggregated_sub_output, self.aggregated_weight, self.aggregated_bias)

        o = self.recurrent_activation(aggregated_input)

        h = o * self.activation(c)

        return h, [h, c] + new_sub_states

class TKAN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        sub_kan_configs: Optional[List[Union[int, float, dict, str]]] = None,
        sub_kan_output_dim: Optional[int] = None,
        sub_kan_input_dim: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.,
        bidirectional: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.tkan_cells = nn.ModuleList()
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                cell = TKANCell(
                    input_size if layer == 0 else hidden_size * self.num_directions,
                    hidden_size,
                    sub_kan_configs=sub_kan_configs,
                    sub_kan_output_dim=sub_kan_output_dim,
                    sub_kan_input_dim=sub_kan_input_dim,
                    use_bias=bias,
                    dropout=dropout if layer > 0 else 0,
                    **kwargs
                )
                self.tkan_cells.append(cell)

    def forward(self, x, initial_states=None):
        is_packed = isinstance(x, nn.utils.rnn.PackedSequence)
        if is_packed:
            x, batch_sizes = x.data, x.batch_sizes
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(1)

        if initial_states is None:
            initial_states = self.init_hidden(max_batch_size)

        outputs = []
        last_states = []

        for layer in range(self.num_layers):
            if is_packed:
                input_layer = nn.utils.rnn.PackedSequence(x, batch_sizes)
            else:
                input_layer = x

            output_layer = []
            for direction in range(self.num_directions):
                layer_idx = layer * self.num_directions + direction
                cell = self.tkan_cells[layer_idx]
                
                state = initial_states[layer_idx]
                steps = reversed(range(x.size(0))) if direction == 1 else range(x.size(0))
                
                for t in steps:
                    h, state = cell(x[t], state)
                    output_layer.append(h.unsqueeze(0))
                    print(f"[TKAN]\t added output_layer! {len(output_layer)}")
                last_states.append(state)
                print(f"[TKAN] added last state! last_states: {len(last_states)}")
            
            x = torch.cat([o for o in output_layer], dim=2)
            outputs.append(x)

        output = outputs[-1]
        if is_packed:
            output = nn.utils.rnn.PackedSequence(output, batch_sizes)
        print(f"[TKAN]\treturning output! {output.shape}, last_states: {len(last_states)}")
        return output, last_states

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return [
            [
                weight.new(batch_size, self.hidden_size).zero_(),  # h
                weight.new(batch_size, self.hidden_size).zero_(),  # c
            ] + [
                weight.new(batch_size, self.tkan_cells[0].sub_kan_output_dim).zero_()
                for _ in range(len(self.tkan_cells[0].sub_kan_configs))
            ]
            for _ in range(self.num_layers * self.num_directions)
        ] 