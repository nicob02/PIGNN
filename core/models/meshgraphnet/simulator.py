from .model import EncoderProcesserDecoder
import torch.nn as nn
import torch
from torch_geometric.data import Data
from core.utils.gnnutils import copy_geometric_data
import os

class Simulator(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, 
                 ndim, device, model_dir='checkpoint/simulator.pth') -> None:
        super(Simulator, self).__init__()

        self.node_input_size =  node_input_size
        self.edge_input_size = edge_input_size
        self.model_dir = model_dir
        self.ndim = ndim
        self.model = EncoderProcesserDecoder(message_passing_num=message_passing_num, 
                                             node_input_size=node_input_size,
                                             edge_input_size=edge_input_size, 
                                             ndim=ndim).to(device)

        self.device = device
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.uniform_(m.bias.data, b=0.001)


    def forward(self, graph:Data, **argv):
        
        
        # Create noise tensors for temperature and voltage
        temp_noise = torch.normal(mean=0.5, std=0.1, size=(graph.x.shape[0], 1), device=graph.x.device)  # Noise ~ N(0, 1)
        volt_noise = torch.normal(mean=0.1, std=0.05, size=(graph.x.shape[0], 1), device=graph.x.device)  # Noise ~ N(0, 0.5)

        # Add noise to the temperature (1st column) and voltage (2nd column)
        graph.x[:, 0:1] += temp_noise 
        graph.x[:, 1:2] += volt_noise  
    
        graph_last = copy_geometric_data(graph)
        node_type = torch.squeeze(graph.node_type).clone()
        one_hot = torch.nn.functional.one_hot(node_type, 3)
        graph.x = torch.cat([graph.x, one_hot], dim=-1)   

        if torch.isnan(graph.x).any():
            print("Warning: NaN detected in graph.x after concatenation in Simulator")
            
        predicted = self.model(graph)  
    

        v = predicted[:, :self.ndim] + graph_last.x[:, :self.ndim] # temp and volt values corresponds to the first two columns of predicted matrix.
        
        if torch.isnan(v).any():
            print("Warning: NaN detected in final output v in Simulator")
            
        return v
    
    def save_model(self, optimizer=None):
        path = os.path.dirname(self.model_dir)
        if not os.path.exists(path):
            os.makedirs(path)

        optimizer_dict = {}
        optimizer_dict.update({'optimizer': optimizer.state_dict()})    # Learning rate/optimization params
            
        to_save_dict ={'model':self.state_dict()}   # Model's weight params
        to_save_dict.update(optimizer_dict)
        
        torch.save(to_save_dict, self.model_dir)
        
    def load_model(self, model_dir=None, optimizer=None):

        if model_dir is None:
            model_dir = self.model_dir
        
        tmp = torch.load(model_dir, map_location='cpu')
        # print(tmp)
        dicts = tmp['model']
        self.load_state_dict(dicts, strict=True)
        
        if optimizer is None: return        
        optimizer.load_state_dict(tmp['optimizer'])
        

            
