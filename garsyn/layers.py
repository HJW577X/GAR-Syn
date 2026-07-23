import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, JumpingKnowledge, global_max_pool


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        y = self.fc2(F.gelu(self.fc1(x)))
        y = self.dropout(y)
        return self.norm(x + y)


class DrugGraphEncoder(nn.Module):
    def __init__(self, num_layers, hidden_dim):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.jump = JumpingKnowledge("cat")
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = 77 if layer == 0 else hidden_dim
            block = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(block))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, graphs):
        batch_graph = graphs if isinstance(graphs, Batch) else Batch.from_data_list(graphs)
        x, edge_index, batch = batch_graph.x, batch_graph.edge_index, batch_graph.batch
        states = []
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(conv(x, edge_index))
            x = norm(x)
            states.append(x)
        node_repr = self.jump(states)
        return global_max_pool(node_repr, batch)


class InteractionTypeBias(nn.Module):
    def __init__(self, num_heads, num_interaction_types=5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_heads, num_interaction_types))

    def forward(self, type_matrix):
        return self.bias[:, type_matrix].unsqueeze(0)


class GatedInteractionAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        dropout=0.1,
        use_interaction_type_bias=True,
        use_gate=True,
        num_interaction_types=5,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_interaction_type_bias = use_interaction_type_bias
        self.use_gate = use_gate
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.gate_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.gate_norm = nn.LayerNorm(hidden_size)
        self.out_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.interaction_bias = (
            InteractionTypeBias(num_heads, num_interaction_types)
            if use_interaction_type_bias
            else None
        )

    def forward(self, x, type_matrix=None, return_attn=True):
        batch_size, num_tokens, hidden_size = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if self.interaction_bias is not None and type_matrix is not None:
            logits = logits + self.interaction_bias(type_matrix)

        attn = self.dropout(F.softmax(logits, dim=-1))
        attn_out = torch.matmul(attn, v)

        if self.use_gate:
            gate = torch.sigmoid(self.gate_proj(self.gate_norm(x)))
        else:
            gate = torch.ones_like(x)
        gate = gate.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        gated = attn_out * gate.transpose(1, 2)
        gated = gated.transpose(1, 2).contiguous().view(batch_size, num_tokens, hidden_size)
        out = self.out_norm(x + self.dropout(self.out_proj(gated)))

        if return_attn:
            return out, {"attn": attn, "gate": gate}
        return out


class SynergyAttentionResidual(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def pool(h):
        return h[:, 0, :]

    def forward(self, previous_states):
        if len(previous_states) == 1:
            batch_size = previous_states[0].shape[0]
            alpha = previous_states[0].new_ones(batch_size, 1)
            return previous_states[0], alpha

        query = self.query(self.pool(previous_states[-1]))
        keys = torch.stack([self.key_norm(self.pool(h)) for h in previous_states], dim=1)
        scores = (query.unsqueeze(1) * keys).sum(dim=-1) / math.sqrt(query.shape[-1])
        alpha = F.softmax(scores, dim=-1)
        stacked = torch.stack(previous_states, dim=1)
        residual = (alpha.unsqueeze(-1).unsqueeze(-1) * stacked).sum(dim=1)
        return self.dropout(residual), alpha


class GARBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        dropout,
        use_interaction_type_bias=True,
        use_gate=True,
        use_depth_residual=True,
        num_interaction_types=5,
    ):
        super().__init__()
        self.use_depth_residual = use_depth_residual
        self.attn_residual = SynergyAttentionResidual(hidden_size, dropout)
        self.gated_attention = GatedInteractionAttention(
            hidden_size,
            num_heads,
            dropout,
            use_interaction_type_bias=use_interaction_type_bias,
            use_gate=use_gate,
            num_interaction_types=num_interaction_types,
        )
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)

    def forward(self, previous_states, type_matrix=None, return_aux=True):
        if self.use_depth_residual:
            residual_state, depth_alpha = self.attn_residual(previous_states)
        else:
            residual_state = previous_states[-1]
            depth_alpha = residual_state.new_zeros(residual_state.shape[0], len(previous_states))
            depth_alpha[:, -1] = 1.0
        if return_aux:
            attn_out, attn_aux = self.gated_attention(residual_state, type_matrix, return_attn=True)
            h_new = self.ffn(attn_out)
            return h_new, {
                "attn": attn_aux["attn"],
                "gate": attn_aux["gate"],
                "depth_alpha": depth_alpha,
            }
        attn_out = self.gated_attention(residual_state, type_matrix, return_attn=False)
        return self.ffn(attn_out), None


class GARTransformer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_layers,
        dropout,
        ffn_mult=2,
        use_interaction_type_bias=True,
        use_gate=True,
        use_depth_residual=True,
        num_interaction_types=5,
    ):
        super().__init__()
        intermediate_size = hidden_size * ffn_mult
        self.blocks = nn.ModuleList(
            [
                GARBlock(
                    hidden_size,
                    num_heads,
                    intermediate_size,
                    dropout,
                    use_interaction_type_bias=use_interaction_type_bias,
                    use_gate=use_gate,
                    use_depth_residual=use_depth_residual,
                    num_interaction_types=num_interaction_types,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, type_matrix=None, return_aux=True):
        previous_states = [x]
        if return_aux:
            all_aux = {"attn": [], "gate": [], "depth_alpha": []}

        for block in self.blocks:
            h_new, aux = block(previous_states, type_matrix=type_matrix, return_aux=return_aux)
            previous_states.append(h_new)
            if return_aux:
                all_aux["attn"].append(aux["attn"])
                all_aux["gate"].append(aux["gate"])
                all_aux["depth_alpha"].append(aux["depth_alpha"])

        if return_aux:
            return previous_states[-1], all_aux
        return previous_states[-1]
