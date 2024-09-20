# E(n) Equivariant Graph Neural Networks

This is my attempt to implement [E(n) Equivariant Graph Neural Networks](https://arxiv.org/pdf/2102.09844). E(n) equivariant graph neural networks maintain equivariance to the E(n) symmetry group.

## Installation

To run this project, you'll need to set up a Python environment and install the necessary dependencies.

### Prerequisites

Make sure you have Python 3.10 or higher installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/maxnygma/equivariant-gnn.git
   cd equivariant-gnn
   
2. Install the requirements
    ```bash
    pip install -r requirements.txt
    
## Running the Code

To run the code, you need to generate data and train a model.

### Generating Data

1. Generate N-Body Dynamical System
    ```bash
    python3 data/nbody_simulation.py --save_path nbody_3000
    
2. Download QM9
    ```bash
    python3 data/download_qm9.py --save_path qm9
    
### Running Training

1. Run GNN on N-Body dataset
    ```bash
    python3 main.py --config configs/nbody_gnn.yml
    
2. Run EGNN on N-Body dataset
    ```bash
    python3 main.py --config configs/nbody_egnn.yml
    
3. Run GNN on QM9 dataset
    ```bash
    python3 main.py --config configs/qm9_gnn.yml
    
4. Run EGNN on QM9 dataset
    ```bash
    python3 main.py --config configs/qm9_egnn.yml
    
### Dummy Example

    import torch
    from model import GNN, EGNN
    
    # GNN
    gnn = GNN(node_dim=3, edge_dim=2, hidden_dim=64, num_nodes=4, output_dim=1, num_blocks=3, infer_edges=False,
                prediction_type='default_pool')
    
    # EGNN
    egnn = EGNN(node_dim=3, coords_dim=2, vel_dim=1, edge_dim=2, hidden_dim=64, num_nodes=4, output_dim=1,
                num_blocks=3, infer_edges=True, update_coords=True, use_velocity=True, prediction_type='default_pool')
    
    # 4 nodes with 3 features each
    node_features = torch.tensor([[0.43, 0.2, 0.1], 
                            [0.411, 0.65, 0.99], 
                            [0.23, 0.91, 0.44],
                            [0.85, 0.20, 0.11]], dtype=torch.float32)
    
    # 3 edges
    edge_indices = torch.tensor([[0, 1, 2], 
                                    [1, 2, 3]], dtype=torch.long)
    
    # Each edge has 2 features
    edge_features = torch.tensor([[0.15, 0.53], 
                                    [0.25, 0.66], 
                                    [0.21, 0.44]], dtype=torch.float32)
    
    out_gnn = gnn(node_features, edge_indices, edge_features)
    out_egnn = egnn(node_features, edge_indices, edge_features)
    
    # tensor([0.0704], grad_fn=<SqueezeBackward1>)
    # tensor([0.1396], grad_fn=<SqueezeBackward1>)
