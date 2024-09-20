import os
import torch
import numpy as np
from torch.utils.data import Dataset


class NBodyDataset(Dataset):
    def __init__(self, files_path='/kaggle/working/', split='train', num_samples=None, time_start=30, time_end=40):
        self.time_start = time_start
        self.time_end = time_end
        
        if split not in ['train', 'val', 'test']:
            raise ValueError('Unknown dataset split. Use train, val or test')
        
        loc_complete = np.load(os.path.join(files_path, f'loc_{split}.npy'))
        vel_complete = np.load(os.path.join(files_path, f'vel_{split}.npy'))
        edges_complete = np.load(os.path.join(files_path, f'edges_{split}.npy'))
        
        if num_samples is not None:
            loc_complete = loc_complete[:num_samples, :, :, :]
            vel_complete = vel_complete[:num_samples, :, :, :]
            edges_complete = edges_complete[:num_samples, :, :]
        
        start, end = [], []
        edge_features = []
        for i in range(loc_complete.shape[3]):
            for j in range(loc_complete.shape[3]):
                if i != j:
                    start.append(i)
                    end.append(j)
                    edge_features.append(edges_complete[:, i, j]) # charge as an edge attribute
        
        self.loc = torch.tensor(loc_complete)
        self.vel = torch.tensor(vel_complete)
        self.edges = torch.stack([torch.tensor(start), torch.tensor(end)]) 
        self.edge_features = torch.tensor(np.array(edge_features)).transpose(0, 1).unsqueeze(-1)
    
    def __getitem__(self, idx):
        loc, vel, edge_features = self.loc[idx], self.vel[idx], self.edge_features[idx]
    
        start_loc = loc[self.time_start]
        start_vel = vel[self.time_start]
        
        sample = torch.vstack([start_loc, start_vel]).transpose(0, 1), self.edges, edge_features
        label = loc[self.time_end]
        
        return sample, label
    
    def __len__(self):
        return len(self.loc)
    
    
def prepare_batch(batch):
    samples, labels = batch
    if labels.ndim == 2:
        labels = labels.reshape(-1, labels.shape[1])
    
    nodes, edge_indices, edge_features = samples
    
    for i in range(len(nodes)):
        edge_indices[i] += nodes.shape[1] * i
    
    nodes = nodes.reshape(-1, nodes.shape[-1])
    edge_features = edge_features.reshape(-1, edge_features.shape[-1])
    edge_indices = edge_indices.reshape(2, -1)
    
    return (nodes.float(), edge_indices, edge_features.float()), labels.float()