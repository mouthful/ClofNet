import torch
from torch import nn
import numpy as np
import logging
from models.gcl import Clof_GCL
from .layers import GaussianLayer

class ClofNet(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4,
        coords_weight=1.0, recurrent=True, norm_diff=True, tanh=False,
    ):
        super(ClofNet, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_edge = nn.Sequential(nn.Linear(in_edge_nf, 8), act_fn)

        edge_embed_dim = 10
        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_embed_dim, self.hidden_nf // 2), act_fn,
            nn.Linear(self.hidden_nf // 2, self.hidden_nf // 2), act_fn)

        self.norm_diff = norm_diff
        for i in range(0, self.n_layers):
            self.add_module(
                "gcl_%d" % i,
                Clof_GCL(
                    input_nf=self.hidden_nf,
                    output_nf=self.hidden_nf,
                    hidden_nf=self.hidden_nf,
                    edges_in_d=self.hidden_nf // 2,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    coords_weight=coords_weight,
                    norm_diff=norm_diff,
                    tanh=tanh,
                ),
            )
        self.to(self.device)
        self.params = self.__str__()

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Network Size', params)
        logging.info('Network Size {}'.format(params))
        return str(params)

    def coord2localframe(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col])
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
            cross_norm = (torch.sqrt(
                torch.sum((coord_cross)**2, 1).unsqueeze(1))) + 1
            coord_cross = coord_cross / cross_norm
        coord_vertical = torch.cross(coord_diff, coord_cross)
        return coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)

    def scalarization(self, edges, x):
        coord_diff, coord_cross, coord_vertical = self.coord2localframe(edges, x)
        # Geometric Vectors Scalarization
        row, col = edges
        edge_basis = torch.cat([coord_diff, coord_cross, coord_vertical], dim=1) 
        r_i = x[row]  
        r_j = x[col]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)  
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)  
        # Calculate angle information in local frames
        coff_mul = coff_i * coff_j  # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) + 1e-5
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) + 1e-5
        pesudo_cos = coff_mul.sum(dim=-1, keepdim=True) / coff_i_norm / coff_j_norm
        pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        pesudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        coff_feat = torch.cat([pesudo_angle, coff_i, coff_j], dim=-1)
        return coff_feat

    def forward(self, h, x, edges, vel, edge_attr, node_attr=None, n_nodes=5):
        h = self.embedding_node(h)
        x = x.reshape(-1, n_nodes, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)
        coff_feat = self.scalarization(edges, x_center)
        edge_feat = torch.cat([edge_attr, coff_feat], dim=-1)
        edge_feat = self.fuse_edge(edge_feat)

        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](
                h, edges, x_center, vel, edge_attr=edge_feat, node_attr=node_attr)

        x = x_center.reshape(-1, n_nodes, 3) + centroid
        x = x.reshape(-1, 3)
        return x


class ClofNet_vel(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4,
        coords_weight=1.0, recurrent=True, norm_diff=True, tanh=False,
    ):
        super(ClofNet_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)

        edge_embed_dim = 16
        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_embed_dim, self.hidden_nf // 2), act_fn,
            nn.Linear(self.hidden_nf // 2, self.hidden_nf // 2), act_fn)

        self.norm_diff = True
        for i in range(0, self.n_layers):
            self.add_module(
                "gcl_%d" % i,
                Clof_GCL(
                    input_nf=self.hidden_nf,
                    output_nf=self.hidden_nf,
                    hidden_nf=self.hidden_nf,
                    edges_in_d=self.hidden_nf // 2,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    coords_weight=coords_weight,
                    norm_diff=norm_diff,
                    tanh=tanh,
                ),
            )
        self.to(self.device)
        self.params = self.__str__()

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Network Size', params)
        logging.info('Network Size {}'.format(params))
        return str(params)

    def coord2localframe(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col])
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
            cross_norm = (torch.sqrt(
                torch.sum((coord_cross)**2, 1).unsqueeze(1))) + 1
            coord_cross = coord_cross / cross_norm
        coord_vertical = torch.cross(coord_diff, coord_cross)
        return coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)

    def scalarization(self, edges, x, vel):
        coord_diff, coord_cross, coord_vertical = self.coord2localframe(edges, x)
        # Geometric Vectors Scalarization
        row, col = edges
        edge_basis = torch.cat([coord_diff, coord_cross, coord_vertical], dim=1) 
        r_i = x[row] 
        r_j = x[col]
        v_i = vel[row]
        v_j = vel[col]
        coff_i = torch.matmul(edge_basis,
                              r_i.unsqueeze(-1)).squeeze(-1)  
        coff_j = torch.matmul(edge_basis,
                              r_j.unsqueeze(-1)).squeeze(-1)  
        vel_i = torch.matmul(edge_basis,
                             v_i.unsqueeze(-1)).squeeze(-1)  
        vel_j = torch.matmul(edge_basis,
                             v_j.unsqueeze(-1)).squeeze(-1)  
        # Calculate angle information in local frames
        coff_mul = coff_i * coff_j  # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True)
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True)
        pesudo_cos = coff_mul.sum(
            dim=-1, keepdim=True) / (coff_i_norm + 1e-5) / (coff_j_norm + 1e-5)
        pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        pesudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        coff_feat = torch.cat([pesudo_angle, coff_i, coff_j, vel_i, vel_j],
                              dim=-1)  #[E, 14]
        return coff_feat

    def forward(self, h, x, edges, vel, edge_attr, node_attr=None, n_nodes=5):
        h = self.embedding_node(h)
        x = x.reshape(-1, n_nodes, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)

        coff_feat = self.scalarization(edges, x_center, vel)
        edge_feat = torch.cat([edge_attr, coff_feat], dim=-1)
        edge_feat = self.fuse_edge(edge_feat)

        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](
                h, edges, x_center, vel, edge_attr=edge_feat, node_attr=node_attr)

        x = x_center.reshape(-1, n_nodes, 3) + centroid
        x = x.reshape(-1, 3)
        return x


class ClofNet_vel_gbf(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4,
        coords_weight=1.0, recurrent=True, norm_diff=True, tanh=False,
    ):
        super(ClofNet_vel_gbf, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)
        self.gbf = GaussianLayer(K=self.hidden_nf // 2, edge_types=8)
        edge_embed_dim = 14
        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_embed_dim, self.hidden_nf // 2), act_fn,
            nn.Linear(self.hidden_nf // 2, self.hidden_nf // 2), act_fn)

        self.norm_diff = True
        for i in range(0, self.n_layers):
            self.add_module(
                "gcl_%d" % i,
                Clof_GCL(
                    input_nf=self.hidden_nf,
                    output_nf=self.hidden_nf,
                    hidden_nf=self.hidden_nf,
                    edges_in_d=self.hidden_nf // 2,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    coords_weight=coords_weight,
                    norm_diff=norm_diff,
                    tanh=tanh,
                ),
            )
        self.to(self.device)
        self.params = self.__str__()

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Network Size', params)
        logging.info('Network Size {}'.format(params))
        return str(params)

    def coord2localframe(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col])
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
            cross_norm = (torch.sqrt(
                torch.sum((coord_cross)**2, 1).unsqueeze(1))) + 1
            coord_cross = coord_cross / cross_norm
        coord_vertical = torch.cross(coord_diff, coord_cross)
        return coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)

    def embed_edge(self, edge_types, dist):
        edge_types = edge_types * 0.5 + 0.5
        return self.gbf(dist, edge_types.long())

    def scalarization(self, edges, x, vel):
        coord_diff, coord_cross, coord_vertical = self.coord2localframe(edges, x)
        # Geometric Vectors Scalarization
        row, col = edges
        edge_basis = torch.cat([coord_diff, coord_cross, coord_vertical], dim=1) 
        r_i = x[row] 
        r_j = x[col]
        v_i = vel[row]
        v_j = vel[col]
        coff_i = torch.matmul(edge_basis,
                              r_i.unsqueeze(-1)).squeeze(-1)  
        coff_j = torch.matmul(edge_basis,
                              r_j.unsqueeze(-1)).squeeze(-1)  
        vel_i = torch.matmul(edge_basis,
                             v_i.unsqueeze(-1)).squeeze(-1)  
        vel_j = torch.matmul(edge_basis,
                             v_j.unsqueeze(-1)).squeeze(-1)  
        # Calculate angle information in local frames
        coff_mul = coff_i * coff_j  # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True)
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True)
        pesudo_cos = coff_mul.sum(
            dim=-1, keepdim=True) / (coff_i_norm + 1e-5) / (coff_j_norm + 1e-5)
        pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        pesudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        coff_feat = torch.cat([pesudo_angle, coff_i, coff_j, vel_i, vel_j],
                              dim=-1)  #[E, 14]
        return coff_feat

    def forward(self, h, x, edges, vel, edge_attr, node_attr=None, n_nodes=5):

        h = self.embedding_node(h)
        x = x.reshape(-1, n_nodes, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)

        coff_feat = self.scalarization(edges, x_center, vel)
        # edge_feat = torch.cat([edge_attr, coff_feat], dim=-1)
        edge_embed = self.embed_edge(edge_attr[:, 0], edge_attr[:, 1])
        edge_feat = self.fuse_edge(coff_feat)
        edge_feat = edge_feat + edge_embed
        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](
                h, edges, x_center, vel, edge_attr=edge_feat, node_attr=node_attr)

        x = x_center.reshape(-1, n_nodes, 3) + centroid
        x = x.reshape(-1, 3)
        return x

