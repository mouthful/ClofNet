import torch
from torch import nn
from models.gcl import EVFN_GCL_norm, EVFN_GCL, EVFN_GCL_norm
import numpy as np
import logging

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=4.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=True)

    def forward(self, x):
        x_proj = x * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class EVFN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device='cpu', 
        act_fn=nn.SiLU(), 
        n_layers=4,
        coords_weight=1.0,
        recurrent=True,
        norm_diff=False,
        tanh=False,
        n_points=5,
    ):
        super(EVFN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)
        # self.embedding_edge = nn.Linear(in_edge_nf, self.hidden_nf)
        self.embedding_edge = nn.Sequential(
            nn.Linear(in_edge_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        # self.dist_fourier = GaussianFourierProjection(
        #     embedding_size=self.hidden_nf // 2)
        # self.dist_mlp = nn.Linear(self.hidden_nf, self.hidden_nf)
        self.coff_fourier = GaussianFourierProjection(
            embedding_size=self.hidden_nf // 2)
        self.coff_mlp = nn.Linear(3 * self.hidden_nf, self.hidden_nf)
        self.n_points = n_points
        self.project = nn.Sequential(
            nn.Linear(hidden_nf * 2 + 2, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )
        self.norm_diff = True
        for i in range(0, self.n_layers):
            self.add_module(
                "gcl_%d" % i,
                EVFN_GCL(
                    input_nf=self.hidden_nf,
                    output_nf=self.hidden_nf,
                    hidden_nf=self.hidden_nf,
                    edges_in_d=self.hidden_nf,
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

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col])
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
            cross_norm = (
                torch.sqrt(torch.sum((coord_cross)**2, 1).unsqueeze(1))) + 1
            coord_cross = coord_cross / cross_norm

        coord_vertical = torch.cross(coord_diff, coord_cross)

        return coord_diff, coord_cross, coord_vertical

    def get_embedding(self, coff):
        coff_embeds = []
        for i in range(3):
            coff_embeds.append(self.coff_fourier(coff[:, i:i + 1]))

        coff_embeds = torch.cat(coff_embeds, dim=-1)  # [E, 6C]
        coff_embeds = self.coff_mlp(coff_embeds)

        return coff_embeds

    def scalarization(self, edges, x):
        coord_diff, coord_cross, coord_vertical = self.coord2radial(edges, x)
        # Geometric Vectors Scalarization
        row, col = edges
        # N, B, C = config.n_points, config.batch_size, self.hidden_nf
        edge_basis = torch.cat(
            [
                coord_diff.unsqueeze(1),
                coord_cross.unsqueeze(1),
                coord_vertical.unsqueeze(1),
            ],
            dim=1,
        )  # [B*N*(N-1), 3]
        r_i = x[row]  # [B*N*(N-1), 3]
        r_j = x[col]
        coff_i = torch.matmul(edge_basis,
                              r_i.unsqueeze(-1)).squeeze(-1)  # [B*N*(N-1), 3]
        coff_j = torch.matmul(edge_basis,
                              r_j.unsqueeze(-1)).squeeze(-1)  # [B*N*(N-1), 3]

        coff_mul = coff_i * coff_j  # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True)
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True)
        pesudo_cos = coff_mul.sum(
            dim=-1, keepdim=True) / (coff_i_norm + 1e-5) / (coff_j_norm + 1e-5)
        pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        pesudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        embed_i = self.get_embedding(coff_i)
        embed_j = self.get_embedding(coff_j)
        edge_embed = torch.cat([pesudo_angle, embed_i, embed_j], dim=-1)
        edge_embed = self.project(edge_embed)
        return edge_embed

    def forward(self, h, x, edges, vel, edge_attr, node_attr=None):
        h = self.embedding_node(h)
        x = x.reshape(-1, self.n_points, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)

        edge_embed = self.scalarization(edges, x_center)
        edge_feature = self.embedding_edge(edge_attr)
        edge_feat = edge_embed * edge_feature

        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](h, edges, x_center, vel, edge_attr=edge_feat, node_attr=node_attr)

        x = x_center.reshape(-1, self.n_points, 3) + centroid
        x = x.reshape(-1, 3)
        return x

class EVFN_naive(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device='cpu', 
        act_fn=nn.SiLU(), 
        n_layers=4,
        coords_weight=1.0,
        recurrent=True,
        norm_diff=False,
        tanh=False,
        n_points=5,
    ):
        super(EVFN_naive, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)
        # self.embedding_edge = nn.Linear(in_edge_nf, self.hidden_nf)
        self.embedding_edge = nn.Sequential(
            nn.Linear(in_edge_nf, 8),
            act_fn)

        edge_embed_dim = 10
        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_embed_dim, self.hidden_nf//2),
            act_fn,
            nn.Linear(self.hidden_nf//2, self.hidden_nf//2),
            act_fn)

        self.n_points = n_points
        self.norm_diff = True
        for i in range(0, self.n_layers):
            self.add_module(
                "gcl_%d" % i,
                EVFN_GCL(
                    input_nf=self.hidden_nf,
                    output_nf=self.hidden_nf,
                    hidden_nf=self.hidden_nf,
                    edges_in_d=self.hidden_nf//2,
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

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col])
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
            cross_norm = (
                torch.sqrt(torch.sum((coord_cross)**2, 1).unsqueeze(1))) + 1
            coord_cross = coord_cross / cross_norm

        coord_vertical = torch.cross(coord_diff, coord_cross)

        return coord_diff, coord_cross, coord_vertical

    def scalarization(self, edges, x):
        coord_diff, coord_cross, coord_vertical = self.coord2radial(edges, x)
        # Geometric Vectors Scalarization
        row, col = edges
        # N, B, C = config.n_points, config.batch_size, self.hidden_nf
        edge_basis = torch.cat(
            [
                coord_diff.unsqueeze(1),
                coord_cross.unsqueeze(1),
                coord_vertical.unsqueeze(1),
            ],
            dim=1,
        )  # [B*N*(N-1), 3]
        r_i = x[row]  # [B*N*(N-1), 3]
        r_j = x[col]
        coff_i = torch.matmul(edge_basis,
                              r_i.unsqueeze(-1)).squeeze(-1)  # [B*N*(N-1), 3]
        coff_j = torch.matmul(edge_basis,
                              r_j.unsqueeze(-1)).squeeze(-1)  # [B*N*(N-1), 3]

        coff_mul = coff_i * coff_j  # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True)
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True)
        pesudo_cos = coff_mul.sum(
            dim=-1, keepdim=True) / (coff_i_norm + 1e-5) / (coff_j_norm + 1e-5)
        pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        pesudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        coff_feat = torch.cat([pesudo_angle, coff_i, coff_j], dim=-1)
        return coff_feat

    def forward(self, h, x, edges, vel, edge_attr, node_attr=None):
        h = self.embedding_node(h)
        # [BN, 3]
        x = x.reshape(-1, self.n_points, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)

        coff_feat = self.scalarization(edges, x_center)
        edge_feat = torch.cat([edge_attr, coff_feat], dim=-1)
        edge_feat = self.fuse_edge(edge_feat)
        
        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](h, edges, x_center, vel, edge_attr=edge_feat, node_attr=node_attr)
        
        x = x_center.reshape(-1, self.n_points, 3) + centroid
        x = x.reshape(-1, 3)
        return x


class EVFN_norm(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device='cpu', 
        act_fn=nn.SiLU(), 
        n_layers=4,
        coords_weight=1.0,
        recurrent=True,
        norm_diff=False,
        tanh=False,
        n_points=5,
    ):
        super(EVFN_norm, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)
        # self.embedding_edge = nn.Linear(in_edge_nf, self.hidden_nf)
        self.embedding_edge = nn.Sequential(
            nn.Linear(in_edge_nf, 8),
            act_fn)

        edge_embed_dim = 10
        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_embed_dim, self.hidden_nf//2),
            act_fn,
            nn.Linear(self.hidden_nf//2, self.hidden_nf//2),
            act_fn)

        self.n_points = n_points
        self.norm_diff = True
        for i in range(0, self.n_layers):
            self.add_module(
                "gcl_%d" % i,
                EVFN_GCL_norm(
                    input_nf=self.hidden_nf,
                    output_nf=self.hidden_nf,
                    hidden_nf=self.hidden_nf,
                    edges_in_d=self.hidden_nf//2,
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

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col])
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
            cross_norm = (
                torch.sqrt(torch.sum((coord_cross)**2, 1).unsqueeze(1))) + 1
            coord_cross = coord_cross / cross_norm

        coord_vertical = torch.cross(coord_diff, coord_cross)

        return coord_diff, coord_cross, coord_vertical

    def scalarization(self, edges, x):
        coord_diff, coord_cross, coord_vertical = self.coord2radial(edges, x)
        # Geometric Vectors Scalarization
        row, col = edges
        # N, B, C = config.n_points, config.batch_size, self.hidden_nf
        edge_basis = torch.cat(
            [
                coord_diff.unsqueeze(1),
                coord_cross.unsqueeze(1),
                coord_vertical.unsqueeze(1),
            ],
            dim=1,
        )  # [B*N*(N-1), 3]
        r_i = x[row]  # [B*N*(N-1), 3]
        r_j = x[col]
        coff_i = torch.matmul(edge_basis,
                              r_i.unsqueeze(-1)).squeeze(-1)  # [B*N*(N-1), 3]
        coff_j = torch.matmul(edge_basis,
                              r_j.unsqueeze(-1)).squeeze(-1)  # [B*N*(N-1), 3]

        coff_mul = coff_i * coff_j  # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True)
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True)
        pesudo_cos = coff_mul.sum(
            dim=-1, keepdim=True) / (coff_i_norm + 1e-5) / (coff_j_norm + 1e-5)
        pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        pesudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        coff_feat = torch.cat([pesudo_angle, coff_i, coff_j], dim=-1)
        return coff_feat

    def forward(self, h, x, edges, vel, edge_attr, node_attr=None):
        h = self.embedding_node(h)
        # [BN, 3]
        x = x.reshape(-1, self.n_points, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)

        coff_feat = self.scalarization(edges, x_center)
        edge_feat = torch.cat([edge_attr, coff_feat], dim=-1)
        edge_feat = self.fuse_edge(edge_feat)
        
        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](h, edges, x_center, vel, edge_attr=edge_feat, node_attr=node_attr)
        
        x = x_center.reshape(-1, self.n_points, 3) + centroid
        x = x.reshape(-1, 3)
        return x


class EVFN_vel(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device='cpu', 
        act_fn=nn.SiLU(), 
        n_layers=4,
        coords_weight=1.0,
        recurrent=True,
        norm_diff=False,
        tanh=False,
        n_points=5,
    ):
        super(EVFN_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)
        # self.embedding_edge = nn.Linear(in_edge_nf, self.hidden_nf)
        # self.embedding_edge = nn.Sequential(
        #     nn.Linear(in_edge_nf, 8),
        #     act_fn)

        edge_embed_dim = 16
        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_embed_dim, self.hidden_nf//2),
            act_fn,
            nn.Linear(self.hidden_nf//2, self.hidden_nf//2),
            act_fn)

        self.n_points = n_points
        self.norm_diff = True
        for i in range(0, self.n_layers):
            self.add_module(
                "gcl_%d" % i,
                EVFN_GCL_norm(
                    input_nf=self.hidden_nf,
                    output_nf=self.hidden_nf,
                    hidden_nf=self.hidden_nf,
                    edges_in_d=self.hidden_nf//2,
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

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col])
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
            cross_norm = (
                torch.sqrt(torch.sum((coord_cross)**2, 1).unsqueeze(1))) + 1
            coord_cross = coord_cross / cross_norm

        coord_vertical = torch.cross(coord_diff, coord_cross)

        return coord_diff, coord_cross, coord_vertical

    def scalarization(self, edges, x, vel):
        coord_diff, coord_cross, coord_vertical = self.coord2radial(edges, x)
        # Geometric Vectors Scalarization
        row, col = edges
        # N, B, C = config.n_points, config.batch_size, self.hidden_nf
        edge_basis = torch.cat(
            [
                coord_diff.unsqueeze(1),
                coord_cross.unsqueeze(1),
                coord_vertical.unsqueeze(1),
            ],
            dim=1,
        )  # [B*N*(N-1), 3]
        r_i = x[row]  # [B*N*(N-1), 3]
        r_j = x[col]
        v_i = vel[row]
        v_j = vel[col]
        coff_i = torch.matmul(edge_basis,
                              r_i.unsqueeze(-1)).squeeze(-1)  # [B*N*(N-1), 3]
        coff_j = torch.matmul(edge_basis,
                              r_j.unsqueeze(-1)).squeeze(-1)  # [B*N*(N-1), 3]
        vel_i = torch.matmul(edge_basis,
                              v_i.unsqueeze(-1)).squeeze(-1)  # [B*N*(N-1), 3]
        vel_j = torch.matmul(edge_basis,
                              v_j.unsqueeze(-1)).squeeze(-1)  # [B*N*(N-1), 3]

        coff_mul = coff_i * coff_j  # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True)
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True)
        pesudo_cos = coff_mul.sum(
            dim=-1, keepdim=True) / (coff_i_norm + 1e-5) / (coff_j_norm + 1e-5)
        pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        pesudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        coff_feat = torch.cat([pesudo_angle, coff_i, coff_j, vel_i, vel_j], dim=-1) #[E, 14]
        return coff_feat

    def forward(self, h, x, edges, vel, edge_attr, node_attr=None):
        h = self.embedding_node(h)
        # [BN, 3]
        x = x.reshape(-1, self.n_points, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)

        coff_feat = self.scalarization(edges, x_center, vel)
        edge_feat = torch.cat([edge_attr, coff_feat], dim=-1)
        edge_feat = self.fuse_edge(edge_feat)
        
        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](h, edges, x_center, vel, edge_attr=edge_feat, node_attr=node_attr)
        
        x = x_center.reshape(-1, self.n_points, 3) + centroid
        x = x.reshape(-1, 3)
        return x

class EVFN_modular(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device='cpu', 
        act_fn=nn.SiLU(), 
        n_layers=4,
        coords_weight=1.0,
        recurrent=True,
        norm_diff=False,
        tanh=False,
        n_points=5,
    ):
        super(EVFN_modular, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)
        # self.embedding_edge = nn.Linear(in_edge_nf, self.hidden_nf)
        self.embedding_edge = nn.Sequential(
            nn.Linear(in_edge_nf, 8),
            act_fn)

        edge_embed_dim = 10
        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_embed_dim, self.hidden_nf//2),
            act_fn,
            nn.Linear(self.hidden_nf//2, self.hidden_nf//2),
            act_fn)

        self.n_points = n_points
        self.norm_diff = True
        for i in range(0, self.n_layers):
            self.add_module(
                "gcl_%d" % i,
                EVFN_GCL_norm(
                    input_nf=self.hidden_nf,
                    output_nf=self.hidden_nf,
                    hidden_nf=self.hidden_nf,
                    edges_in_d=self.hidden_nf//2,
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

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col])
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
            cross_norm = (
                torch.sqrt(torch.sum((coord_cross)**2, 1).unsqueeze(1))) + 1
            coord_cross = coord_cross / cross_norm

        coord_vertical = torch.cross(coord_diff, coord_cross)

        return coord_diff, coord_cross, coord_vertical

    def scalarization(self, edges, x):
        coord_diff, coord_cross, coord_vertical = self.coord2radial(edges, x)
        # Geometric Vectors Scalarization
        row, col = edges
        # N, B, C = config.n_points, config.batch_size, self.hidden_nf
        edge_basis = torch.cat(
            [
                coord_diff.unsqueeze(1),
                coord_cross.unsqueeze(1),
                coord_vertical.unsqueeze(1),
            ],
            dim=1,
        )  # [B*N*(N-1), 3]
        r_i = x[row]  # [B*N*(N-1), 3]
        r_j = x[col]
        coff_i = torch.matmul(edge_basis,
                              r_i.unsqueeze(-1)).squeeze(-1)  # [B*N*(N-1), 3]
        coff_j = torch.matmul(edge_basis,
                              r_j.unsqueeze(-1)).squeeze(-1)  # [B*N*(N-1), 3]

        coff_mul = coff_i * coff_j  # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) + 1e-5
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) + 1e-5
        coff_i = coff_i / coff_i_norm
        coff_j = coff_j / coff_j_norm
        pesudo_cos = coff_mul.sum(
            dim=-1, keepdim=True) / coff_i_norm / coff_j_norm
        pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        pesudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        coff_feat = torch.cat([pesudo_angle, coff_i, coff_j], dim=-1)
        return coff_feat

    def forward(self, h, x, edges, vel, edge_attr, node_attr=None):
        h = self.embedding_node(h)
        # [BN, 3]
        x = x.reshape(-1, self.n_points, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)

        coff_feat = self.scalarization(edges, x_center)
        edge_feat = torch.cat([edge_attr, coff_feat], dim=-1)
        edge_feat = self.fuse_edge(edge_feat)
        
        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](h, edges, x_center, vel, edge_attr=edge_feat, node_attr=node_attr)
        
        x = x_center.reshape(-1, self.n_points, 3) + centroid
        x = x.reshape(-1, 3)
        return x


