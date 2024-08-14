import torch
import torch.nn as nn
import torch.nn.functional as F

from .tkan import TKAN

class AddAndNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.norm_layer = nn.LayerNorm(size)  # None will be replaced with actual size during forward pass
    
    def forward(self, inputs):
        return self.norm_layer(torch.add(*inputs))

class Gate(nn.Module):
    def __init__(self, in_size, out_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size if out_size else in_size
        
        self.dense_layer = nn.Linear(in_size, self.out_size)
        self.gated_layer = nn.Linear(in_size, self.out_size)

    def forward(self, inputs):
        dense_output = self.dense_layer(inputs)
        gated_output = torch.sigmoid(self.gated_layer(inputs))
        return dense_output * gated_output

class GRN(nn.Module):
    def __init__(self, in_size, hidden_size=None, out_size=None):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size if hidden_size else in_size
        self.out_size = out_size if out_size else self.hidden_size
        print(f"[GRN] in_size: {self.in_size}, hidden_size: {self.hidden_size}, out_size: {self.out_size}")
        self.skip_layer = nn.Linear(self.in_size, self.out_size)
        self.hidden_layer_1 = nn.Sequential(
            nn.Linear(self.in_size, self.hidden_size),
            nn.ELU()
        )
        self.hidden_layer_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.gate_layer = Gate(self.hidden_size, out_size=self.out_size)
        self.add_and_norm_layer = AddAndNorm(self.out_size)

    def forward(self, x):
        skip = self.skip_layer(x)
        hidden = self.hidden_layer_1(x)
        hidden = self.hidden_layer_2(hidden)
        gating_output = self.gate_layer(hidden)
        return self.add_and_norm_layer([skip, gating_output])

class VariableSelectionNetwork(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 time_steps, 
                 embedding_dim, 
                 num_features, 
                 name=""
    ):
        super().__init__()
        self.name = name if name else VariableSelectionNetwork.get_name()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.flatten_dim = embedding_dim * num_features
        self.mlp_dense = GRN(self.flatten_dim, 
                             hidden_size=self.hidden_size, 
                             out_size=num_features)
        self.grn_layers = nn.ModuleList([
            GRN(1, hidden_size=self.hidden_size) 
            for _ in range(num_features)
        ])

    @classmethod
    def get_name(cls):
        counter = getattr(cls, "counter", 0) + 1
        setattr(cls, "counter", counter)
        return counter

    def forward(self, x):        
        print(f"[VariableSelectionNetwork] [{self.name}]\tinput shape: {x.shape}")
        batch_size, time_steps, embedding_dim, num_inputs = x.size()
        flatten = x.view(batch_size, time_steps, embedding_dim * num_inputs)
        print(f"[VariableSelectionNetwork] [{self.name}-\tflatten shape: {flatten.shape}")
        
        mlp_outputs = self.mlp_dense(flatten)
        sparse_weights = F.softmax(mlp_outputs, dim=-1).unsqueeze(2)
        print(f"[VariableSelectionNetwork] [{self.name}-\tsparse_weights shape: {sparse_weights.shape}")
        
        trans_emb_list = [
            self.grn_layers[i](x[:, :, :, i]) 
            for i in range(num_inputs)
        ]
        print(f"[VariableSelectionNetwork] [{self.name}]\tlen emb list: {len(trans_emb_list)}")
        transformed_embedding = torch.stack(trans_emb_list, dim=-1)
        print(f"[VariableSelectionNetwork] [{self.name}]\ttransformed_embedding shape: {transformed_embedding.shape}")
        
        combined = sparse_weights * transformed_embedding        
        temporal_ctx = torch.sum(combined, dim=-1)
        print(f"[VariableSelectionNetwork] [{self.name}]\toutput shape: {temporal_ctx.shape}")
        
        return temporal_ctx

class RecurrentLayer(nn.Module):
    def __init__(self, in_size, return_state=False, use_tkan=False, name=""):
        super().__init__()
        self.name = name if name else RecurrentLayer.get_name()
        self.return_state = return_state
        if use_tkan:
            self.layer = TKAN(in_size, in_size, return_sequences=True, return_state=return_state)
        else:
            self.layer = nn.LSTM(in_size, in_size, batch_first=True)
    
    @classmethod
    def get_name(cls):
        counter = getattr(cls, "counter", 0) + 1
        setattr(cls, "counter", counter)
        return counter
    
    def forward(self, x, initial_state=None):
        print(f"[RecurrentLayer] [{self.name}]\t input shape: {x.shape}")
        if initial_state is not None:
            if isinstance(initial_state, (tuple, list)):
                if len(initial_state) == 1:
                    initial_state = initial_state[0]
                if isinstance(initial_state, (tuple, list)):
                    for idx, i in enumerate(initial_state):
                        if i is not None:
                            print(f"[RecurrentLayer] [{self.name}]\t initial_state {idx} shape: {[s.shape for s in i]}")
                        else:
                            print(f"[RecurrentLayer] [{self.name}]\t initial_state {idx} shape: None")
            if not isinstance(initial_state, (tuple, list)):
                print(f"[RecurrentLayer] [{self.name}]\t initial_state shape: {[s.shape for s in initial_state]}")
        output, hidden = self.layer(x, initial_state)
        if isinstance(hidden, (tuple, list)):
            if len(hidden) == 2:
                (h_n, c_n) = hidden
            elif len(hidden) == 1:
                h_n = hidden[0]
                c_n = None
            else:
                h_n = None
                c_n = None
        if self.return_state:            
            return output, h_n, c_n
        else:            
            return output

class EmbeddingLayer(nn.Module):
    def __init__(self, num_embedding, num_features):
        super().__init__()
        self.num_features = num_features
        
        self.dense_layers = nn.ModuleList([
            nn.Linear(1, num_embedding) 
            for _ in range(num_features)
        ])

    def forward(self, x):
        x_split = torch.split(x, (1,) * self.num_features, -1)
        
        embeddings = [
            dense_layer(x_slice) 
            for dense_layer, x_slice
            in zip(
                self.dense_layers, 
                x_split
            )
        ]
        out = torch.stack(embeddings, dim=-1)
        print(f"[EmbeddingLayer] Output shape: {out.shape}")
        return out

class TKAT(nn.Module):
    """Temporal Kan Transformer model

    Args:
        sequence_length (int): length of past sequence to use
        num_unknow_features (int): number of observed features (can be 0)
        num_know_features (int): number of known features (if 0 then create futures values for recurrent decoder)
        num_embedding (int): size of the embedding
        num_hidden (int): number of hidden units in layers
        num_heads (int): number of heads for multihead attention
        n_ahead (int): number of steps to predict
        use_tkan (bool, optional): Wether or not to use TKAN instead of LSTM. Defaults to True.
    """
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
        self.n_known_features = num_known_features
        self.n_unknown_features = num_unknown_features
        self.n_features = num_unknown_features + num_known_features
        self.n_ahead = n_ahead
        self.input_size = self.n_features + self.n_ahead
        
        self.embedding_layer = EmbeddingLayer(num_embedding,self.n_features)
        self.vsn_past_features = VariableSelectionNetwork(num_hidden, 
                                                          sequence_length, 
                                                          num_embedding, 
                                                          self.n_features,
                                                          name="past_features")
        self.vsn_future_features = VariableSelectionNetwork(num_hidden, 
                                                            n_ahead, 
                                                            num_embedding, 
                                                            num_known_features,
                                                            name="future_features")
        
        self.encoder = RecurrentLayer(num_hidden, 
                                      return_state=True, 
                                      use_tkan=use_tkan,
                                      name="encoder")
        self.decoder = RecurrentLayer(num_hidden, 
                                      return_state=False, 
                                      use_tkan=use_tkan,
                                      name="decoder")
        
        self.gate = Gate(num_hidden)
        self.add_and_norm = AddAndNorm(num_hidden)
        self.grn = GRN(num_hidden)
        
        self.attention = nn.MultiheadAttention(num_hidden, num_heads, batch_first=True)
        
        self.final_dense = nn.Linear(sequence_length * (sequence_length + n_ahead) * num_hidden, n_ahead)

    def forward(self, x):
        print(f"[TKAT] Input shape: {x.shape}")
        embedded_inputs = self.embedding_layer(x)
        
        past_features = embedded_inputs[:, :self.sequence_length, :, :]
        future_features = embedded_inputs[:, self.sequence_length:, :, -self.n_known_features:]
        print(f"[TKAT] past_features shape: {past_features.shape}")
        print(f"[TKAT] future_features shape: {future_features.shape}")
        variable_selection_past = self.vsn_past_features(past_features)
        variable_selection_future = self.vsn_future_features(future_features)
        print(f"[TKAT] variable_selection_past shape: {variable_selection_past.shape}")
        print(f"[TKAT] variable_selection_future shape: {variable_selection_future.shape}")
        
        encode_out, *encode_states = self.encoder(variable_selection_past)
        print(f"[TKAT] encode_out shape: {encode_out.shape}")
        print(f"[TKAT] encode_states shape: {len(encode_states)}, state: {[s.shape for s in encode_states[0]]}")
        decode_out = self.decoder(variable_selection_future, initial_state=encode_states)
        
        history = torch.cat([encode_out, decode_out], dim=1)
        
        selected = torch.cat([variable_selection_past, variable_selection_future], dim=1)
        all_context = self.add_and_norm([self.gate(history), selected])
        
        enriched = self.grn(all_context)
        
        attention_output, _ = self.attention(enriched, enriched, enriched)
        
        flattened_output = attention_output.flatten(start_dim=1)
        dense_output = self.final_dense(flattened_output)
        
        return dense_output
