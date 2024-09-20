import torch
import torch.nn as nn

from model.model_utils import aggregate_message, OutputMLP
    

class GraphConvolution(nn.Module):
    def __init__(self, edge_func, node_func, infer_edges_func, infer_edges):
        super(GraphConvolution, self).__init__()
        
        self.edge_func = edge_func
        self.node_func = node_func
        self.infer_edges_func = infer_edges_func
        
        self.infer_edges = infer_edges
    
    def forward(self, inputs):
        node_features, edge_indices, edge_features = inputs
        start, end = edge_indices
       
        x = torch.cat([node_features[start], node_features[end], edge_features], dim=1)
        
        message = self.edge_func(x)
        
        if self.infer_edges:
            inferred_edges = self.infer_edges_func(message)
            message = message * inferred_edges

        agg_message = aggregate_message(message, start, node_features.shape[0])
        x = torch.cat([node_features, agg_message], dim=1)
    
        node_features += self.node_func(x)
        
        return node_features, edge_indices, edge_features
    

class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_nodes, output_dim=1, num_blocks=1, infer_edges=False,
                 prediction_type='default_pool'):
        super(GNN, self).__init__()
        
        self.node_embeddings = nn.Linear(node_dim, hidden_dim)
        
        convs = []
        for _ in range(num_blocks):
            edge_func = nn.Sequential(
                nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU()
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
            
            _conv = GraphConvolution(edge_func, node_func, infer_edges_func, infer_edges)
            convs.append(_conv)
            
        self.convs = nn.Sequential(*convs)
        
        self.output_mlp = OutputMLP(hidden_dim, output_dim, num_nodes, prediction_type)
    
    def forward(self, nodes, edge_indices, edge_features):
        nodes_enc = self.node_embeddings(nodes)
        nodes_enc, _, _ = self.convs((nodes_enc, edge_indices, edge_features))
        output = self.output_mlp(nodes_enc)
        
        return output
            