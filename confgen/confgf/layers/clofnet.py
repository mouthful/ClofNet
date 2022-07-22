# coding=utf-8
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

from .common import MeanReadout, SumReadout, MultiLayerPerceptron
from .gin import GINEConv
from .gat import Transformer_layer

class EquiLayer(MessagePassing):

    def __init__(self, eps: float = 0., train_eps: bool = False,
                 activation="softplus", **kwargs):
        super(EquiLayer, self).__init__(aggr='add', **kwargs)
        self.initial_eps = eps

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None       

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            # assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor: 
        if self.activation:
            return self.activation(x_j + edge_attr)
        else:
            # return x_j + edge_attr
            return edge_attr

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GradientGCN(torch.nn.Module):

    def __init__(self, hidden_dim, hidden_coff_dim=64, num_convs=3, activation="softplus", readout="sum", short_cut=False, concat_hidden=False):
        super(GradientGCN, self).__init__()

        self.hidden_dim = hidden_dim
        # self.num_convs = num_convs
        self.num_layers = 2
        self.num_convs = 2
        self.short_cut = short_cut
        self.num_head = 8
        self.dropout = 0.1
        self.concat_hidden = concat_hidden
        self.hidden_coff_dim = hidden_coff_dim

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None 
        
        # self.conv_modules = nn.ModuleList()
        self.transformers = nn.ModuleList()
        self.equi_modules = nn.ModuleList()
        self.dynamic_mlp_modules = nn.ModuleList()
        for _ in range(self.num_layers):
            trans_convs = nn.ModuleList()
            for i in range(self.num_convs):
                trans_convs.append(
                    Transformer_layer(self.num_head, self.hidden_dim, dropout=self.dropout, activation=activation)
                )
            # self.conv_modules.append(convs)
            self.transformers.append(trans_convs)
            self.equi_modules.append(EquiLayer(activation=False))
            self.dynamic_mlp_modules.append(
                nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_coff_dim),
                nn.Softplus(),
                nn.Linear(self.hidden_coff_dim, 3))
            )


    def coord2basis(self, data):
        coord_diff = data.pert_pos[data.edge_index[0]] - data.pert_pos[data.edge_index[1]]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(data.pert_pos[data.edge_index[0]], data.pert_pos[data.edge_index[1]])

        norm = torch.sqrt(radial) + 1
        coord_diff = coord_diff / norm
        cross_norm = torch.sqrt(torch.sum((coord_cross)**2, 1).unsqueeze(1)) + 1
        coord_cross = coord_cross / cross_norm
        
        coord_vertical = torch.cross(coord_diff, coord_cross)

        return coord_diff, coord_cross, coord_vertical

    
    def forward(self, data, node_attr, edge_attr):
        """
        Input:
            data: (torch_geometric.data.Data): batched graph
            node_attr: node feature tensor with shape (num_node, hidden)
            edge_attr: edge feature tensor with shape (num_edge, hidden)
        Output:
            node_attr
            graph feature
        """
 
        hiddens = []
        conv_input = node_attr # (num_node, hidden)

        for module_idx, convs in enumerate(self.transformers):
            for conv_idx, conv in enumerate(convs):
                hidden = conv(data.edge_index, conv_input, edge_attr)
                if conv_idx < len(convs) - 1 and self.activation is not None:
                    hidden = self.activation(hidden)
                assert hidden.shape == conv_input.shape                
                if self.short_cut and hidden.shape == conv_input.shape:
                    hidden += conv_input

                hiddens.append(hidden)
                conv_input = hidden

            if self.concat_hidden:
                node_feature = torch.cat(hiddens, dim=-1)
            else:
                node_feature = hiddens[-1]

            h_row, h_col = node_feature[data.edge_index[0]], node_feature[data.edge_index[1]] # (num_edge, hidden)
            edge_feature = torch.cat([h_row*h_col, edge_attr], dim=-1) # (num_edge, 2 * hidden)
            ## generate gradient
            dynamic_coff = self.dynamic_mlp_modules[module_idx](edge_feature)
            coord_diff, coord_cross, coord_vertical = self.coord2basis(data)
            basis_mix = dynamic_coff[:, :1] * coord_diff + dynamic_coff[:, 1:2] * coord_cross + dynamic_coff[:, 2:3] * coord_vertical
            
            if module_idx == 0:
                gradient = self.equi_modules[module_idx](node_feature, data.edge_index, basis_mix)
            else:
                gradient += self.equi_modules[module_idx](node_feature, data.edge_index, basis_mix)

        return {
            "node_feature": node_feature,
            "gradient": gradient
        }

