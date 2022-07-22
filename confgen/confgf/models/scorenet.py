import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from confgf import utils, layers


class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class EquiDistanceScoreMatch(torch.nn.Module):

    def __init__(self, config):
        super(EquiDistanceScoreMatch, self).__init__()
        self.config = config
        self.anneal_power = self.config.train.anneal_power
        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.noise_type = self.config.model.noise_type

        self.node_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.edge_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.dist_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.input_mlp = layers.MultiLayerPerceptron(2 * self.hidden_dim, [self.hidden_dim], activation=self.config.model.mlp_act)
        self.coff_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.coff_mlp = nn.Linear(4 * self.hidden_dim, self. hidden_dim)
        self.project = layers.MultiLayerPerceptron(2 * self.hidden_dim + 2, [self.hidden_dim, self.hidden_dim], activation=self.config.model.mlp_act)

        self.model = layers.GradientGCN(hidden_dim=self.hidden_dim, hidden_coff_dim=128, \
                                 num_convs=self.config.model.num_convs, \
                                 activation=self.config.model.gnn_act, \
                                 readout="sum", short_cut=self.config.model.short_cut, \
                                 concat_hidden=self.config.model.concat_hidden)
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_noise_level)), dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False) # (num_noise_level)
        """
        Techniques from "Improved Techniques for Training Score-Based Generative Models"
        1. Choose sigma1 to be as large as the maximum Euclidean distance between all pairs of training data points.
        2. Choose sigmas as a geometric progression with common ratio gamma, where a specific equation of CDF is satisfied.
        3. Parameterize the Noise Conditional Score Networks with f_theta_sigma(x) =  f_theta(x) / sigma
        """

    
    @torch.no_grad()
    # extend the edge on the fly, second order: angle, third order: dihedral
    def extend_graph(self, data: Data, order=3):

        def binarize(x):
            return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

        def get_higher_order_adj_matrix(adj, order):
            """
            Args:
                adj:        (N, N)
                type_mat:   (N, N)
            """
            adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                        binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

            for i in range(2, order+1):
                adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
            order_mat = torch.zeros_like(adj)

            for i in range(1, order+1):
                order_mat += (adj_mats[i] - adj_mats[i-1]) * i

            return order_mat

        num_types = len(utils.BOND_TYPES)

        N = data.num_nodes
        adj = to_dense_adj(data.edge_index).squeeze(0)
        adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

        type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(0)   # (N, N)
        type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_type = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.bond_edge_index = data.edge_index  # Save original edges
        data.edge_index, data.edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N) # modify data
        edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
        data.is_bond = (data.edge_type < num_types)
        assert (data.edge_index == edge_index_1).all()

        return data

    # @torch.no_grad()
    def get_distance(self, data: Data):
        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        data.edge_length = d
        return data   

    # @torch.no_grad()
    def get_perturb_distance(self, data: Data, p_pos):
        pos = p_pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        return d  

    def get_pred_distance(self, data: Data, p_pos):
        pos = p_pos
        row, col = data.edge_index
        d = torch.sqrt(torch.sum((pos[row] - pos[col])**2, dim=-1) + 0.0001)
        # d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        return d    

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

    @torch.no_grad()
    def get_angle(self, data: Data, p_pos):
        pos = p_pos
        row, col = data.edge_index
        pos_normal = pos.clone().detach()
        pos_normal_norm = pos_normal.norm(dim=-1).unsqueeze(-1)
        pos_normal = pos_normal / (pos_normal_norm + 1e-5)
        cos_theta = torch.sum(pos_normal[row] * pos_normal[col], dim=-1, keepdim=True)
        sin_theta = torch.sqrt(1 - cos_theta ** 2)
        node_angles = torch.cat([cos_theta, sin_theta], dim=-1)
        return node_angles

    @torch.no_grad()
    def get_score(self, data: Data, d, sigma):
        """
        Input:
            data: torch geometric batched data object
            d: edge distance, shape (num_edge, 1)
            sigma: noise level, tensor (,)
        Output:
            log-likelihood gradient of distance, tensor with shape (num_edge, 1)         
        """
        # generate common features
        node_attr = self.node_emb(data.atom_type) # (num_node, hidden)
        edge_attr = self.edge_emb(data.edge_type) # (num_edge, hidden)
        d_emb = self.dist_gaussian_fourier(d)
        d_emb = self.input_mlp(d_emb) # (num_edge, hidden)
        edge_attr = d_emb * edge_attr # (num_edge, hidden)
        
        # construct geometric features
        row, col = data.edge_index[0], data.edge_index[1] # check if roe and col is right?
        coord_diff, coord_cross, coord_vertical = self.coord2basis(data) # [E, 3]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1) # [E, 3]
        r_i, r_j = data.pert_pos[row], data.pert_pos[col] # [E, 3]
        # [E, 3, 3] x [E, 3, 1]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1) # [E, 3]
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1) # [E, 3]
        coff_mul = coff_i * coff_j # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True)
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True)
        pesudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + 1e-5) / (coff_j_norm + 1e-5)
        pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        psudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        embed_i = self.get_embedding(coff_i) # [E, C]
        embed_j = self.get_embedding(coff_j) # [E, C]
        edge_embed = torch.cat([psudo_angle, embed_i, embed_j], dim=-1)
        edge_embed = self.project(edge_embed)

        edge_attr = edge_attr + edge_embed

        output = self.model(data, node_attr, edge_attr)
        scores = output["gradient"] * (1. / sigma) # f_theta_sigma(x) =  f_theta(x) / sigma, (num_edge, 1)
        return scores

    def get_embedding(self, coff_index):
        coff_embeds = []
        for i in [0, 2]: # if i=1, then x=0
            coff_embeds.append(self.coff_gaussian_fourier(coff_index[:, i:i+1])) #[E, 2C]
        coff_embeds = torch.cat(coff_embeds, dim=-1) # [E, 6C]
        coff_embeds = self.coff_mlp(coff_embeds)
        
        return coff_embeds

    def forward(self, data):
        """
        Input:
            data: torch geometric batched data object
        Output:
            loss
        """
        # a workaround to get the current device, we assume all tensors in a model are on the same device.
        self.device = self.sigmas.device
        data = self.extend_graph(data, self.order)
        ## enable input gradient
        input_x = data.pos
        input_x.requires_grad = True

        data = self.get_distance(data)

        assert data.edge_index.size(1) == data.edge_length.size(0)
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]        

        # sample noise level
        noise_level = torch.randint(0, self.sigmas.size(0), (data.num_graphs,), device=self.device) # (num_graph)
        used_sigmas = self.sigmas[noise_level] # (num_graph)
        used_sigmas = used_sigmas[node2graph].unsqueeze(-1) # (num_nodes, 1)

        if self.noise_type == 'rand':
            coord_noise = torch.randn_like(data.pos) 
        else:
            raise NotImplementedError('noise type must in [distance_symm, distance_rand]')
  
        assert coord_noise.shape == data.pos.shape
        perturbed_pos = data.pos + coord_noise * used_sigmas 
        data.pert_pos = perturbed_pos 
        perturbed_d = self.get_perturb_distance(data, perturbed_pos)
        target = -1 / (used_sigmas ** 2) * (perturbed_pos - data.pos)

        # generate common features
        node_attr = self.node_emb(data.atom_type) # (num_node, hidden)
        edge_attr = self.edge_emb(data.edge_type) # (num_edge, hidden)
        d_emb = self.dist_gaussian_fourier(perturbed_d)
        d_emb = self.input_mlp(d_emb) # (num_edge, hidden)
        edge_attr = d_emb * edge_attr # (num_edge, hidden)
        
        # construct geometric features
        row, col = data.edge_index[0], data.edge_index[1] # check if roe and col is right?
        coord_diff, coord_cross, coord_vertical = self.coord2basis(data) # [E, 3]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1) # [E, 3]
        r_i, r_j = data.pert_pos[row], data.pert_pos[col] # [E, 3]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1) # [E, 3]
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1) # [E, 3]
        coff_mul = coff_i * coff_j # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True)
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True)
        pesudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + 1e-5) / (coff_j_norm + 1e-5)
        pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        psudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        embed_i = self.get_embedding(coff_i) # [E, C]
        embed_j = self.get_embedding(coff_j) # [E, C]
        edge_embed = torch.cat([psudo_angle, embed_i, embed_j], dim=-1)
        edge_embed = self.project(edge_embed)
        edge_attr = edge_attr + edge_embed

        # estimate scores
        output = self.model(data, node_attr, edge_attr)
        scores = output["gradient"] * (1. / used_sigmas)
        loss_pos =  0.5 * torch.sum((scores - target) ** 2, -1) * (used_sigmas.squeeze(-1) ** self.anneal_power) # (num_edge)
        loss_pos = scatter_mean(loss_pos, node2graph) # (num_graph)
        
        loss_dict = {
            'position': loss_pos.mean(),
            'distance': torch.Tensor([0]).to(loss_pos.device),
        }
        return loss_dict
