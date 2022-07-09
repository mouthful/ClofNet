import numpy as np
import torch
import random
import os

class NBodyDataset():
    """
    NBodyDataset:
    {
        small: ES
        static: G+ES
        dynamic: L+ES
    }
    """
    def __init__(self, partition='train', max_samples=1e8, data_root=None, data_mode='small'):
        self.partition = partition
        self.sufix = partition
        if data_mode == 'small':
            self.sufix += "_charged5_initvel1small"
        elif (data_mode == 'static') or (data_mode == 'dynamic'):
            self.sufix += f"_{data_mode}5_initvel1{data_mode}"
        elif data_mode == "small_20body":
            self.sufix += f"_charged20_initvel1{data_mode}"
        else:
            self.sufix += f"_{data_mode[:-7]}20_initvel1{data_mode}"

        self.data_root = data_root
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()
        self.frame_0 = 30
        self.frame_T = 40

    def load(self):
        loc = np.load(os.path.join(self.data_root, 'loc_' + self.sufix + '.npy'))
        vel = np.load(os.path.join(self.data_root, 'vel_' + self.sufix + '.npy'))
        edges = np.load(os.path.join(self.data_root, 'edges_' + self.sufix + '.npy'))
        charges = np.load(os.path.join(self.data_root, 'charges_' + self.sufix + '.npy'))

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges


    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0:self.max_samples]
        edge_attr = []

        #Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2) # swap n_nodes <--> batch_size and add nf dimension

        return torch.Tensor(loc), torch.Tensor(vel), torch.Tensor(edge_attr), edges, torch.Tensor(charges)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()
    
    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]
        return loc[self.frame_0], vel[self.frame_0], edge_attr, charges, loc[self.frame_T]

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges


if __name__ == "__main__":
    NBodyDataset()