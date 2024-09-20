import torch
import torch.nn as nn
import torch.nn.functional as F


def aggregate_coords(message, indices, result_size):
    ''' Coordinates sum with mean aggregation '''
    
    indices = indices.unsqueeze(-1).expand(-1, message.size(1))
    
    agg_message = torch.scatter_add(torch.zeros((result_size, message.shape[1]), device=message.device),
                                    0, indices, message)
    count = torch.scatter_add(torch.zeros((result_size, message.shape[1]), device=message.device), 
                              0, indices, torch.ones_like(message, device=message.device))
    
    return agg_message / count.clamp(min=1)


def aggregate_message(message, indices, result_size):
    ''' Message passing with sum aggregation '''
    
    indices = indices.unsqueeze(-1).expand(-1, message.size(1))
    agg_message = torch.scatter_add(torch.zeros((result_size, message.shape[1]), device=message.device),
                                    0, indices, message)
    
    return agg_message


def join_graph_props(coords, vels, node_features):
    out = node_features
    
    if vels is not None:
        out = torch.cat([vels, out], dim=1)
    if coords is not None:
        out = torch.cat([coords, out], dim=1)
        
    return out 


def split_graph_props(node_features, coords_size, vel_size):
    if coords_size > 0 and vel_size > 0:
        coords = node_features[:, :coords_size]
        velocities = node_features[:, coords_size:coords_size + vel_size]
        features = node_features[:, coords_size + vel_size:]
        
        if coords_size + vel_size == node_features.shape[1]:
            features = torch.norm(velocities, p=2, dim=1).unsqueeze(1).detach()
    elif coords_size > 0 and vel_size == 0:
        coords = node_features[:, :coords_size]
        velocities = None
        features = node_features[:, coords_size:]
    else:
        coords = None
        velocities = None
        features = node_features
        
    return coords, velocities, features    


class OutputMLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_nodes, prediction_type):
        super(OutputMLP, self).__init__()
        
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        
        self.linear_3 = nn.Linear(output_dim, hidden_dim)
        self.linear_4 = nn.Linear(hidden_dim, 1)
        
        self.num_nodes = num_nodes
        self.prediction_type = prediction_type
    
    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
              
        if self.prediction_type == 'embedding':
            return x
        
        x = x.reshape(-1, self.num_nodes, x.shape[-1])
        x = x.sum(dim=1)
        
        x = self.linear_3(x)
        x = F.silu(x)
        out = self.linear_4(x).squeeze(1)
        
        return out