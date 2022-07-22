import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class Transformer_layer(nn.Module):
    def __init__(
        self, n_head, hidden_dim, dropout=0.2, activation="softplus"
    ):
        super(Transformer_layer, self).__init__()

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        assert hidden_dim % n_head == 0
        self.MHA = TransformerConv(
            in_channels=hidden_dim,
            out_channels=int(hidden_dim // n_head),
            heads=n_head,
            dropout=dropout,
            edge_dim=hidden_dim,
        )
        self.FFN = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, edge_index, node_attr, edge_attr):
        x = self.MHA(node_attr, edge_index, edge_attr)
        # e_index, attn_weights = tuple_attn
        node_attr = node_attr + self.norm1(x)
        x = self.FFN(node_attr)
        node_attr = node_attr + self.norm2(x)
        
        return node_attr

