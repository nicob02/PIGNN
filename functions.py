
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
        if volt is None:
            self.volt = np.random.randint(15,25)        #Training with a random voltage source input
        else:
            self.volt = volt                                # Case where voltage is given for evaluation

    def graph_modify(self, graph, value_last, **argv)->None:
        a,b,c,d,e,f,g = self.params

        grad_value = self.gradop(graph, value_last)
        grad_v = grad_value[1]          # Voltage gradient

        # Calculate the squared magnitude of the gradient: |∇v|^2 = v_x^2 + v_y^2
        squared_abs_grad_v = torch.sum(grad_v ** 2, dim=1, keepdim=True)  # Shape (N, 1)
        temp = value_last[:,0:1]        # Temps values at time t
        sigma = f*(1+g*(temp-e))        # Sigma at time t
        q = sigma*squared_abs_grad_v    # Heat at time t

        if torch.isnan(q).any():
            print(f"Warning: NaN detected in predicted after q")

        graph.x = torch.cat((graph.x,q), dim=-1)    # Append new Q value at t to the input

        return graph    

    def init_condition(self, pos):
        
        volt = torch.full_like(pos[:, 0:1], 0.5)    # Create a tensor filled with 1s for the voltage
        #q =  torch.full_like(pos[:, 1:2], 0)      #q
        
        return volt # Concatenate along the last dimension


    def boundary_condition(self, pos):
        
        volt = torch.full_like(pos[:, 0:1], 0.2)  # Create a tensor filled with 310s for the temperature

        return volt    # Concatenate along the last dimension
    
    def electrode_condition(self, pos, values_last, t):
        # Introduce a time-varying sinusoidal voltage source
        # Frequency = 500 kHz
        rf_frequency = 200

        # Voltage: V(t) = V0 * sin(2 * pi * f * t)
        time_var_volt = self.volt * math.sin(2 * torch.pi * rf_frequency * t)
       
        volt = torch.full_like(pos[:, 0:1], time_var_volt)    # Create a tensor filled with input voltage source
        #volt = torch.full_like(pos[:, 1:2], self.volt)    # Create a tensor filled with input voltage source
        return volt
        
    def compute_gradient(self, field, positions):

        grad = torch.autograd.grad(
            outputs=field,  # Scalar field
            inputs=positions,  # Positions of the nodes
            grad_outputs=torch.ones_like(field),  # Vector of ones for chain rule
            create_graph=True,  # Enable higher-order gradients
            retain_graph=True   # Retain graph for further computations
        )[0]
        
        return grad.requires_grad_()

    def compute_laplacian(self, grad, positions):
 
        laplacian = torch.autograd.grad(
            outputs=grad,  # Gradient field
            inputs=positions,  # Positions of the nodes
            grad_outputs=torch.ones_like(grad),  # Vector of ones for chain rule
            create_graph=True,  # Enable higher-order gradients
            allow_unused=True 
        )[0]
        return laplacian.sum(dim=-1, keepdim=True)  # Sum over spatial dimensions
    
    def pde(self, graph, values_last, values_this, **argv):

        a,b,c,d,e,f,g = self.params
              
        volt_last = values_last[:,0:1]
        volt_this = values_this[:,0:1]

        epsilon = 1e-4
  

            
        grad_value = self.gradop(graph, values_this)
        grad_v = grad_value[1]          # Volt Gradient at t+1
        print("graph.pos")
        print(graph.pos)
        print("graph.x")
        print(graph.x)
    
        squared_abs_grad_v = torch.sum(grad_v ** 2, dim=1, keepdim=True)  # Shape (N, 1)
            
        sigma = f*(1+g*(temp_this - e)) # Sigma at t+1

        lap_value = self.laplacianop(graph,volt_this)
    
        lap_volt = lap_value[:,0:1]

        #∇ · (σ(T)∇v) = 0
        loss_volt = sigma*lap_volt
 
        print("lap_volt")
        print(lap_volt)
      
                     
        print("losses_tempthen_volt")
        print(loss_volt)
            
        return loss_volt
        

    
    

    
    
