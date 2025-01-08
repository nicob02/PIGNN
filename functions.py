
import torch
from core.pde import laplacian, grad
import numpy as np
import math

class ElectroThermalFunc(): 

    func_name = 'rfa'
    def __init__(self, delta_t, params) -> None:
        self.delta_t = delta_t
        self.params = params
        self.laplacianop = laplacian()
        self.gradop = grad()
                    

    def graph_modify(self, graph, value_last, **argv)->None:
        
        x = graph.pos[:, 0:1]
        y = graph.pos[:, 1:2]
        freq = self.params

        f = -2*freq*freq*torch.sin(freq*x)*torch.sin(freq*y)
        #f = 4*freq*freq*torch.sin(freq*x)*torch.sin(freq*y)
        graph.x = torch.cat((graph.x,f), dim=-1)    # Append source f(x,y) value to the input

        return graph    

    def init_condition(self, pos):
        
        volt = torch.full_like(pos[:, 0:1], 0)    # Create a tensor filled with 0s for the voltage

        return volt 

    def boundary_condition(self, pos):
        
        volt = torch.full_like(pos[:, 0:1], 0)  # Create a tensor filled with 0s for the B.C. voltage

        return volt   
    

    def pde(self, graph, values_last, values_this, **argv):

        x = graph.pos[:, 0:1]
        y = graph.pos[:, 1:2]
        freq = self.params        # w = angular frequency
              
        volt_last = values_last[:,0:1]
        volt_this = values_this[:,0:1]

        print("graph.pos")
        print(graph.pos)
        print("graph.x")
        print(graph.x)

        lap_value = self.laplacianop(graph,volt_this)
    
        lap_volt = lap_value[:,0:1]

        f = -2*freq*freq*torch.sin(freq*x)*torch.sin(freq*y)
        #f = 4*freq*freq*torch.sin(freq*x)*torch.sin(freq*y)
        
        #-2w^2*sin(w*x)*sin(wy) + ∇ · (∇v) = 0
        loss_volt = f + lap_volt
 
        print("lap_volt")
        print(lap_volt)
      
        print("losses_volt")
        print(loss_volt)
            
        return loss_volt
        

    
    

    
    
