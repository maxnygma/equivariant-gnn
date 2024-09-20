import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import GNN, EGNN
from data import NBodyDataset, QM9Dataset, prepare_batch


class Trainer():
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Device is {self.device}')
         
        if config.task == 'nbody':
            train_dataset = NBodyDataset(files_path=config.data.root, split='train', 
                                         num_samples=config.data.max_num_samples, time_start=config.data.time_start, 
                                         time_end=config.data.time_end)
            self.train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
            
            val_dataset = NBodyDataset(files_path=config.data.root, split='val', 
                                       num_samples=config.data.max_num_samples, time_start=config.data.time_start, 
                                       time_end=config.data.time_end)
            self.val_dataloader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
            
            test_dataset = NBodyDataset(files_path=config.data.root, split='test', 
                                       num_samples=config.data.max_num_samples, time_start=config.data.time_start, 
                                       time_end=config.data.time_end)
            self.test_dataloader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
            
            self.loss_function = F.mse_loss
        elif config.task == 'qm9':
            train_dataset = QM9Dataset(files_path=config.data.root, split='train', 
                                       target_property=config.data.target_property)
            # train_dataset = torch.utils.data.Subset(train_dataset, list(range(100)))
            self.train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
            
            val_dataset = QM9Dataset(files_path=config.data.root, split='valid', 
                                     target_property=config.data.target_property)
            self.val_dataloader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
            
            test_dataset = QM9Dataset(files_path=config.data.root, split='test', 
                                      target_property=config.data.target_property)
            self.test_dataloader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
            
            self.loss_function = F.l1_loss
        else:
            raise NotImplementedError
        
        if config.model.type == 'gnn':
            self.model = GNN(node_dim=config.model.node_dim, edge_dim=config.model.edge_dim, 
                             hidden_dim=config.model.hidden_dim, num_nodes=config.training.num_nodes, 
                             output_dim=config.model.output_dim, num_blocks=config.model.num_blocks,
                             infer_edges=config.model.infer_edges, prediction_type=config.model.prediction_type).to(self.device)
        elif config.model.type == 'egnn':
            self.model = EGNN(node_dim=config.model.node_dim, coords_dim=config.model.coords_dim, 
                 vel_dim=config.model.vel_dim, edge_dim=config.model.edge_dim, hidden_dim=config.model.hidden_dim,
                 num_nodes=config.training.num_nodes, output_dim=config.model.output_dim, 
                 num_blocks=config.model.num_blocks, prediction_type=config.model.prediction_type, 
                 infer_edges=config.model.infer_edges, use_velocity=config.model.use_velocity, 
                 update_coords=config.model.update_coords).to(self.device)
        else:
            raise ValueError('Unknown model type')
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.training.lr, 
                                    weight_decay=config.training.weight_decay)
        
        if config.task == 'qm9':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.training.epochs)
        else:
            self.scheduler = None
        
    def run(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.epochs): 
            train_loss = self.train()
            val_loss = self.validate()
                
            print(f'Epoch {epoch + 1}: train loss {train_loss}, val loss {val_loss}')
            
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                
                torch.save({
                    'epoch': epoch,                        
                    'model_state_dict': self.model.state_dict(),  
                    'optimizer_state_dict': self.optimizer.state_dict(),  
                    'loss': val_loss,                          
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, 'best.ckpt')
            
           
        # Test here
        test_loss = self.test()
        print(f'Test loss {test_loss}')
    
    def train(self):
        self.model.train()
        total_loss = 0.0
        total_norm = 0.0
        
        for step, batch in enumerate(self.train_dataloader): 
            self.optimizer.zero_grad()

            batch = prepare_batch(batch)
            sample, label = batch 
            sample, label = [x.to(self.device) for x in sample], label.to(self.device)
            if label.ndim == 3:
                label = label.reshape(-1, label.shape[1])
            
            nodes, edge_indices, edge_features = sample
            out = self.model(nodes.float(), edge_indices, edge_features.float())  

            train_loss = self.loss_function(out, label)            
            train_loss.backward()

            batch_grad_norm = self.get_grad_norm()
            total_norm += batch_grad_norm

            self.optimizer.step()

            total_loss += train_loss.item()
        
        if self.scheduler is not None:
            self.scheduler.step()
            
        print(f'Grad norm {total_norm / len(self.train_dataloader)}')
        
        return train_loss / len(self.train_dataloader)
            
    def validate(self):
        self.model.eval()
        total_val_loss = 0.0
        
        for step, batch in enumerate(self.val_dataloader):
            batch = prepare_batch(batch)
            sample, label = batch 
            sample, label = [x.to(self.device) for x in sample], label.to(self.device)
            if label.ndim == 3:
                label = label.reshape(-1, label.shape[1])

            nodes, edge_indices, edge_features = sample

            with torch.no_grad():
                out = self.model(nodes.float(), edge_indices, edge_features.float())  

            val_loss = self.loss_function(out, label)  
            total_val_loss += val_loss.item()
            
        return total_val_loss / len(self.val_dataloader)
    
    def test(self):
        self.model.eval()
        total_test_loss = 0.0
        
        for step, batch in enumerate(self.test_dataloader):
            batch = prepare_batch(batch)
            sample, label = batch 
            sample, label = [x.to(self.device) for x in sample], label.to(self.device)
            if label.ndim == 3:
                label = label.reshape(-1, label.shape[1])

            nodes, edge_indices, edge_features = sample

            with torch.no_grad():
                out = self.model(nodes.float(), edge_indices, edge_features.float())  

            test_loss = self.loss_function(out, label)  
            total_test_loss += test_loss.item()
            
        return total_test_loss / len(self.test_dataloader)
    
    def get_grad_norm(self):
        batch_grad_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                batch_grad_norm += param_norm.item() ** 2
        batch_grad_norm  = batch_grad_norm  ** 0.5
        
        return batch_grad_norm