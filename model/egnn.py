import torch
import torch.nn as nn

from model.model_utils import split_graph_props, aggregate_message, aggregate_coords, join_graph_props, OutputMLP


class EquivariantGraphConvolution(nn.Module):
    def __init__(self, edge_func, coords_func, vel_func, node_func, infer_edges_func, infer_edges, 
                 use_velocity, update_coords, coords_size, vel_size):
        super(EquivariantGraphConvolution, self).__init__()
        
        self.edge_func = edge_func
        self.coords_func = coords_func
        self.vel_func = vel_func
        self.node_func = node_func
        self.infer_edges_func = infer_edges_func
        
        self.infer_edges = infer_edges
        self.use_velocity = use_velocity
        self.update_coords = update_coords
        
        self.coords_size = coords_size
        self.vel_size = vel_size
        
    def forward(self, inputs):
        node_features_input, edge_indices, edge_features = inputs
        coords, vels, node_features = split_graph_props(node_features_input, self.coords_size, self.vel_size)
        
        start, end = edge_indices
        
        # Compute coords diff norm
        coords_diff = coords[start] - coords[end]
        coords_diff_norm = torch.norm(coords_diff, p=2, dim=1).unsqueeze(1) 
        
        # Get message
        x = torch.cat([node_features[start], node_features[end], coords_diff_norm, edge_features], dim=1)
        message = self.edge_func(x)
        
        # Update coordinates
        if self.update_coords:
            coords_model_term = coords_diff * self.coords_func(message)
            coords = coords + aggregate_coords(coords_model_term, start, coords.shape[0])
            
            if self.use_velocity:
                coords = coords + self.vel_func(node_features) * vels
                
        # Infer edges
        if self.infer_edges:
            inferred_edges = self.infer_edges_func(message)
            message = message * inferred_edges
                
        # Aggregate message
        agg_message = aggregate_message(message, start, node_features.shape[0])
        
        x = torch.cat([node_features, agg_message], dim=1)
        node_features = node_features + self.node_func(x)
                
        output_features = join_graph_props(coords, vels, node_features)
                
        return (output_features, edge_indices, edge_features)

    
class EGNN(nn.Module):
    def __init__(self, node_dim, coords_dim, vel_dim, edge_dim, hidden_dim, num_nodes,
                 output_dim, num_blocks, infer_edges=False, update_coords=True, use_velocity=True, 
                 prediction_type='default'):
        super(EGNN, self).__init__()
        
        node_dim_ = node_dim - coords_dim - vel_dim
        if node_dim_ == 0:
            node_dim_ = 1
        
        self.node_embeddings = nn.Linear(node_dim_, hidden_dim)
        
        convs = []
        for _ in range(num_blocks):
            edge_func = nn.Sequential(
                nn.Linear(2 * hidden_dim + edge_dim + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU()
            )
            
            coords_func = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, coords_dim)
            )
            
            vel_func = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )
            
            node_func = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            infer_edges_func = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            
            _conv = EquivariantGraphConvolution(edge_func, coords_func, vel_func, node_func, infer_edges_func,
                                                infer_edges, use_velocity, update_coords, coords_dim, vel_dim)
            convs.append(_conv)
            
        self.convs = nn.Sequential(*convs)
        
        self.output_mlp = OutputMLP(hidden_dim, output_dim, num_nodes, prediction_type)
        
        self.coords_size = coords_dim
        self.vel_size = vel_dim
        self.prediction_type = prediction_type
        
    def forward(self, nodes, edge_indices, edge_features):
        coords, vels, node_features = split_graph_props(nodes, self.coords_size, self.vel_size)
        
        nodes_enc = self.node_embeddings(node_features)
        
        nodes_enc, _, _ = self.convs((join_graph_props(coords, vels, nodes_enc), edge_indices, edge_features))
        
        coords, vels, node_features_final = split_graph_props(nodes_enc, self.coords_size, self.vel_size)
        
        if self.prediction_type == 'coords':           
            return coords
        elif self.prediction_type == 'default_pool' or self.prediction_type == 'embeddings':
            output = self.output_mlp(node_features_final)
            
            return output
        else:
            raise ValueError()