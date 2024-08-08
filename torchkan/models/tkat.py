import torch
import torch.nn as nn
import torch.nn.functional as F

from .tkan import TKAN

class AddAndNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.norm_layer = nn.LayerNorm(size)  # None will be replaced with actual size during forward pass
    
    def forward(self, inputs):
        tmp = torch.add(*inputs)
        return self.norm_layer(tmp)

class Gate(nn.Module):
    def __init__(self, hidden_layer_size=None):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        
    def build(self, input_shape):
        print(f"[Gate] build; {input_shape}")
        if self.hidden_layer_size is None:
            self.hidden_layer_size = input_shape[-1]
        self.dense_layer = nn.Linear(input_shape[-1], self.hidden_layer_size)
        self.gated_layer = nn.Linear(input_shape[-1], self.hidden_layer_size)

    def forward(self, inputs):
        if not hasattr(self, 'dense_layer'):
            self.build(inputs.size())
        dense_output = self.dense_layer(inputs)
        gated_output = torch.sigmoid(self.gated_layer(inputs))
        return dense_output * gated_output

class GRN(nn.Module):
    def __init__(self, hidden_layer_size, output_size=None):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size or hidden_layer_size
        
        self.skip_layer = nn.Linear(hidden_layer_size, self.output_size)
        self.hidden_layer_1 = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ELU()
        )
        self.hidden_layer_2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.gate_layer = Gate(self.output_size)
        self.add_and_norm_layer = AddAndNorm(self.output_size)

    def forward(self, inputs):
        skip = self.skip_layer(inputs)
        hidden = self.hidden_layer_1(inputs)
        hidden = self.hidden_layer_2(hidden)
        gating_output = self.gate_layer(hidden)
        return self.add_and_norm_layer([skip, gating_output])

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.num_hidden = num_hidden

    def build(self, input_shape):
        _, time_steps, embedding_dim, num_inputs = input_shape
        self.num_inputs = num_inputs
        self.flatten_dim = time_steps * embedding_dim * num_inputs
        self.mlp_dense = GRN(hidden_layer_size=self.num_hidden, output_size=num_inputs)
        self.grn_layers = nn.ModuleList([GRN(self.num_hidden) for _ in range(num_inputs)])

    def forward(self, inputs):
        if not hasattr(self, 'mlp_dense'):
            self.build(inputs.size())
        
        batch_size, time_steps, embedding_dim, num_inputs = inputs.size()
        flatten = inputs.view(batch_size, time_steps, -1)
        
        mlp_outputs = self.mlp_dense(flatten)
        sparse_weights = F.softmax(mlp_outputs, dim=-1).unsqueeze(2)
        
        trans_emb_list = [self.grn_layers[i](inputs[:, :, :, i]) for i in range(num_inputs)]
        transformed_embedding = torch.stack(trans_emb_list, dim=-1)
        
        combined = sparse_weights * transformed_embedding
        temporal_ctx = torch.sum(combined, dim=-1)
        
        return temporal_ctx

class RecurrentLayer(nn.Module):
    def __init__(self, num_units, return_state=False, use_tkan=False):
        super().__init__()
        self.return_state = return_state
        if use_tkan:
            self.layer = TKAN(num_units, num_units)
        else:
            self.layer = nn.LSTM(num_units, num_units, batch_first=True)

    def forward(self, inputs, initial_state=None):
        if self.return_state:
            output, (h_n, c_n) = self.layer(inputs, initial_state)
            return output, h_n, c_n
        else:
            output, _ = self.layer(inputs, initial_state)
            return output

class EmbeddingLayer(nn.Module):
    def __init__(self, input_size, output_size, num_features):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_features = num_features
        
        self.dense_layers = nn.ModuleList([
            nn.Linear(1, self.output_size) 
            for _ in range(num_features)
        ])

    def forward(self, x):
        shape = x.shape

        embeddings = [dense_layer(x_slice) for i, (dense_layer, x_slice) in enumerate(zip(self.dense_layers, torch.split(x, (1,) * shape[-1], -1)))]
        return torch.stack(embeddings, dim=-1)

class TKAT(nn.Module):
    """ Temporal Kan Transformer """
    def __init__(self, 
                 sequence_length: int, 
                 num_unknown_features: int, 
                 num_known_features: int, 
                 num_embedding: int, 
                 num_hidden: int, 
                 num_heads: int, 
                 n_ahead: int, 
                 use_tkan: bool = True
    ):        
        super().__init__()
        self.sequence_length = sequence_length
        self.n_ahead = n_ahead
        
        self.embedding_layer = EmbeddingLayer(sequence_length, num_hidden, num_embedding)
        self.vsn_past_features = VariableSelectionNetwork(num_hidden)
        self.vsn_future_features = VariableSelectionNetwork(num_hidden)
        
        self.encoder = RecurrentLayer(num_hidden, return_state=True, use_tkan=use_tkan)
        self.decoder = RecurrentLayer(num_hidden, return_state=False, use_tkan=use_tkan)
        
        self.gate = Gate(num_hidden)
        self.add_and_norm = AddAndNorm(num_hidden)
        self.grn = GRN(num_hidden)
        
        self.attention = nn.MultiheadAttention(num_hidden, num_heads, batch_first=True)
        
        self.final_dense = nn.Linear(sequence_length * (sequence_length + n_ahead) * num_hidden, n_ahead)

    def forward(self, x):
        embedded_inputs = self.embedding_layer(x)
        
        past_features = embedded_inputs[:, :self.sequence_length, :, :]
        future_features = embedded_inputs[:, self.sequence_length:, :, -x.shape[-1]:]
        
        variable_selection_past = self.vsn_past_features(past_features)
        variable_selection_future = self.vsn_future_features(future_features)
        
        encode_out, *encode_states = self.encoder(variable_selection_past)
        decode_out = self.decoder(variable_selection_future, initial_state=encode_states)
        
        history = torch.cat([encode_out, decode_out], dim=1)
        
        selected = torch.cat([variable_selection_past, variable_selection_future], dim=1)
        all_context = self.add_and_norm([self.gate(history), selected])
        
        enriched = self.grn(all_context)
        
        attention_output, _ = self.attention(enriched, enriched, enriched)
        
        flattened_output = attention_output.flatten(start_dim=1)
        dense_output = self.final_dense(flattened_output)
        
        return dense_output
