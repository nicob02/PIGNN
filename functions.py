import torch
from core.pde import laplacian, grad
import numpy as np


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
        
        temp = torch.full_like(pos[:, 0:1], 310)  # Create a tensor filled with 310s for the temperature
        volt = torch.full_like(pos[:, 1:2], 0)    # Create a tensor filled with 1s for the voltage
        #q =  torch.full_like(pos[:, 1:2], 0)      #q
        
        return torch.cat((temp, volt), dim=-1)    # Concatenate along the last dimension


    def boundary_condition(self, pos):
        
        temp = torch.full_like(pos[:, 0:1], 310)  # Create a tensor filled with 310s for the temperature
        volt = torch.full_like(pos[:, 1:2], 0)    # Create a tensor filled with 1s for the voltage
    
        return torch.cat((temp, volt), dim=-1)    # Concatenate along the last dimension
    
    def electrode_condition(self, pos, values_last, t):
        

Thought about time-dependent voltage boundary condition durante 4 segundos
Below is an updated version of your code with the key changes in the electrode_condition method to apply a time-varying sinusoidal voltage instead of a DC voltage. In addition, a brief comment has been added to remind you that you may need to choose very small time steps to adequately capture the oscillatory behavior.

Key Changes:

Introduced a radiofrequency sinusoidal voltage boundary condition in the electrode_condition method using the provided time t and a chosen frequency of 500 kHz.
Ensure that the time t passed to electrode_condition is in seconds and that your time stepping is sufficiently fine to resolve the sinusoidal signal.
Modified Code:

python
Copiar código
import torch
from core.pde import laplacian, grad
import numpy as np

class ElectroThermalFunc(): 
    func_name = 'rfa'
    def __init__(self, delta_t, params, volt=None) -> None:
        self.delta_t = delta_t
        self.params = params
        self.laplacianop = laplacian()
        self.gradop = grad()
        if volt is None:
            self.volt = np.random.randint(15,25)  # Training with a random voltage source input
        else:
            self.volt = volt                     # Case where voltage is given for evaluation

    def graph_modify(self, graph, value_last, **argv)->None:
        a,b,c,d,e,f,g = self.params

        grad_value = self.gradop(graph, value_last)
        grad_v = grad_value[1]          # Voltage gradient

        # Calculate the squared magnitude of the gradient: |∇v|^2 = v_x^2 + v_y^2
        squared_abs_grad_v = torch.sum(grad_v ** 2, dim=1, keepdim=True)  # Shape (N, 1)
        temp = value_last[:,0:1]        # Temperature values at time t
        sigma = f*(1+g*(temp-e))        # Temperature-dependent conductivity at time t
        q = sigma*squared_abs_grad_v    # Heat source term (Joule heating) at time t

        if torch.isnan(q).any():
            print(f"Warning: NaN detected in predicted after q")

        graph.x = torch.cat((graph.x,q), dim=-1)    # Append new Q value at t to the input
        return graph    

    def init_condition(self, pos):
        # Initial condition: temperature = 310 K, voltage = 0 V
        temp = torch.full_like(pos[:, 0:1], 310) 
        volt = torch.full_like(pos[:, 1:2], 0)
        return torch.cat((temp, volt), dim=-1)    

    def boundary_condition(self, pos):
        # Boundary condition: temperature = 310 K, voltage = 0 V at domain boundaries
        temp = torch.full_like(pos[:, 0:1], 310)
        volt = torch.full_like(pos[:, 1:2], 0)
        return torch.cat((temp, volt), dim=-1)

    def electrode_condition(self, pos, values_last, t):
        # Introduce a time-varying sinusoidal voltage source
        # Frequency = 500 kHz
        rf_frequency = 500e3

        # Voltage: V(t) = V0 * sin(2 * pi * f * t)
        time_var_volt = self.volt * torch.sin(2 * torch.pi * rf_frequency * t)
        time_var_volt = time_var_volt.expand_as(pos[:, 1:2])  # match shape
        
        temp = values_last[:,0:1]
        #volt = torch.full_like(pos[:, 1:2], self.volt)    # Create a tensor filled with input voltage source
        return torch.cat((temp, time_var_volt), dim=-1)

    
    def pde(self, graph, values_last, values_this, **argv):

        a,b,c,d,e,f,g = self.params
              
        temp_last = values_last[:,0:1]
        volt_last = values_last[:,1:2]
        temp_this = values_this[:,0:1]
        volt_this = values_this[:,1:2]
    
        epsilon = 1e-4
        dvdt = (temp_this-temp_last)/self.delta_t
            
        grad_value = self.gradop(graph, values_this)
        grad_v = grad_value[1]          # Volt Gradient at t+1

        squared_abs_grad_v = torch.sum(grad_v ** 2, dim=1, keepdim=True)  # Shape (N, 1)
        
        sigma = f*(1+g*(temp_this - e)) # Sigma at t+1
        q = sigma*squared_abs_grad_v    # q at t+1

        lap_value = self.laplacianop(graph,values_this)
    
        lap_temp = lap_value[:,0:1]
        lap_volt = lap_value[:,1:2]

        #∇ · (σ(T)∇v) = 0
        loss_volt = sigma*lap_volt
        print("lap_volt")
        print(lap_volt)
        #ρticti*∂T/∂t = Q + ∇ · (d∇T) + H(Tbl − T), when H=0 we have the weak-formulation
        print("lap_temp")
        print(lap_temp)
        loss_temp = (0.01*((a*b*dvdt) - q - c*lap_temp -d*(e-temp_this)))
        
        print("losses_tempthen_volt")
        print(loss_temp)
        print(loss_volt)
            
        return torch.cat([loss_temp,loss_volt],axis=1)

    
    

    
    
