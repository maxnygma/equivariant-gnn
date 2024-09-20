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