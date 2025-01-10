
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
        #freq = self.params
        #f = -2*freq*freq*torch.sin(freq*x)*torch.sin(freq*y)
    
        # f(x,y) = 2π cos(πy) sin(πx)
        #         + 2π cos(πx) sin(πy)
        #         + (x+y) sin(πx) sin(πy)
        #         - 2π² (x+y) sin(πx) sin(πy)
        f = (
            2 * math.pi * torch.cos(math.pi * y) * torch.sin(math.pi * x)
            + 2 * math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
            + (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
            - 2 * (math.pi ** 2) * (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        )

        graph.x = torch.cat((graph.x, f), dim=-1)

        return graph    

    def init_condition(self, pos):
        
        volt = torch.full_like(pos[:, 0:1], 0)    # Create a tensor filled with 0s for the voltage

        return volt 

    def boundary_condition(self, graph, predicted):
        
        volt = torch.full_like(graph.pos[:, 0:1], 0)  # Create a tensor filled with 0s for the B.C. voltage
        return volt   

    def exact_solution(self, graph):
        x = graph.pos[:, 0:1]
        y = graph.pos[:, 1:2]
        
        # Compute (x + y) * sin(pi*x) * sin(pi*y)
        u = (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        
        return u

    def pde(self, graph, values_last, values_this, **argv):

        """
        PDE: -Δu + u = f(x,y)
        with ε=1, k=1, and
        f(x,y) = 2π cos(πy) sin(πx)
               + 2π cos(πx) sin(πy)
               + (x + y) sin(πx) sin(πy)
               - 2 π² (x + y) sin(πx) sin(πy).
        """
    
        # Extract node positions
        x = graph.pos[:, 0:1]
        y = graph.pos[:, 1:2]
    
        # "values_this" is our predicted u at the current iteration
        volt_this = values_this[:, 0:1]
    
        # Compute the Laplacian of volt_this
        lap_value = self.laplacianop(graph, volt_this)
        lap_volt = lap_value[:, 0:1]
    
        # Define the forcing function f(x,y)
        f = (
            2 * math.pi * torch.cos(math.pi * y) * torch.sin(math.pi * x)
            + 2 * math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
            + (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
            - 2 * (math.pi ** 2) * (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        )
    
        # PDE residual:  -Δu + u - f = 0
        # so the "loss" (residual) is
        loss_volt = -lap_volt + volt_this - f
    
        # Optional: print statements for debugging
        print("graph.pos")
        print(graph.pos)
        print("graph.x")
        print(graph.x)
        print("lap_volt")
        print(lap_volt)
        print("losses_volt")
        print(loss_volt)
                
        return loss_volt
        

    
    

    
    
