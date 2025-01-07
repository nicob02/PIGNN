
import torch
from core.pde import laplacian, grad
import numpy as np
import math

class ElectroThermalFunc(): 

    func_name = 'rfa'
    def __init__(self, delta_t, params, volt=None) -> None:
        self.delta_t = delta_t
        self.params = params
        self.laplacianop = laplacian()
        self.gradop = grad()
                    

    def graph_modify(self, graph, value_last, **argv)->None:
        
        x = graph.pos[:, 0:1]
        y = graph.pos[:, 1:2]
        freq = self.params

        f = -2*freq*torch.sin(freq*x)*torch.sin(freq*y)
        graph.x = torch.cat((graph.x,f), dim=-1)    # Append new Q value at t to the input

        return graph    

    def init_condition(self, pos):
        
        volt = torch.full_like(pos[:, 0:1], 0.2)    # Create a tensor filled with 1s for the voltage
        #q =  torch.full_like(pos[:, 1:2], 0)      #q
        
        return volt # Concatenate along the last dimension


    def boundary_condition(self, pos):
        
        volt = torch.full_like(pos[:, 0:1], 0.2)  # Create a tensor filled with 310s for the temperature

        return volt    # Concatenate along the last dimension
    

    def pde(self, graph, values_last, values_this, **argv):

        a,b,c,d,e,f,g = self.params
              
        volt_last = values_last[:,0:1]
        volt_this = values_this[:,0:1]

        epsilon = 1e-4
  

            
        grad_value = self.gradop(graph, values_this)
        grad_v = grad_value[0]          # Volt Gradient at t+1
        print("graph.pos")
        print(graph.pos)
        print("graph.x")
        print(graph.x)
    
        squared_abs_grad_v = torch.sum(grad_v ** 2, dim=1, keepdim=True)  # Shape (N, 1)
            
        sigma = f

        lap_value = self.laplacianop(graph,volt_this)
    
        lap_volt = lap_value[:,0:1]

        #∇ · (σ(T)∇v) = 0
        loss_volt = sigma*lap_volt
 
        print("lap_volt")
        print(lap_volt)
      
                     
        print("losses_volt")
        print(loss_volt)
            
        return loss_volt
        

    
    

    
    
