import numpy as np

import torch
from torch.utils.data import Dataset

class QM9Dataset(Dataset):
    def __init__(self, files_path='/kaggle/working/', split='train', target_property='homo'):
        data = np.load(f'{files_path}/{split}.npz')
        
        labels = data[target_property]
        positions = data['positions']
        charges = data['charges']
        
        atom_features = np.eye(charges.max() + 1, dtype=np.float64)[charges][:, :, np.unique(charges)]
        
        start, end = [], []
        for i in range(positions.shape[1]):
            for j in range(positions.shape[1]):
                if i != j:
                    start.append(i)
                    end.append(j)
        
        self.positions = torch.tensor(positions)
        self.edges = torch.stack([torch.tensor(start), torch.tensor(end)])
        self.atom_features = torch.tensor(atom_features)
        self.labels = torch.tensor(labels)
        
    def __getitem__(self, idx):
        position, atom_feature = self.positions[idx], self.atom_features[idx]
        
        sample = torch.hstack([position, atom_feature]), self.edges, \
                 torch.ones((self.edges[0].shape[0], 1))
        label = self.labels[idx]
        
        return sample, label
    
    def __len__(self):
        return len(self.positions)