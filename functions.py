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

        volt = torch.full_like(pos[:, 1:2], self.volt)    # Create a tensor filled with input voltage source
        temp = values_last[:,0:1]
        
        return torch.cat((temp,volt),dim=-1)
    
    @classmethod
    def exact_solution(cls, pos, t):
        return cls.boundary_condition(pos, t)   #THIS I HAVE TO CHANGE WHEN I KNOW HOW TO GET THE ANSWER NUMERICALLY
    
    def pde(self, graph, values_last, values_this, **argv):

        a,b,c,d,e,f,g = self.params
        values_last = torch.abs(values_last)
        values_this = torch.abs(values_this)

        max_temp = 3000  # Maximum realistic temperature
        min_temp = 310  # Minimum realistic temperature (e.g., absolute zero)
        min_volt = 0
        max_volt = 100
       
        
        temp_last = values_last[:,0:1]  # Temp at time t
        volt_last = values_last[:,1:2]  # Volt at time t
        temp_this = values_this[:,0:1]  # Temp at time t+1
        volt_this = values_this[:,1:2]  # Volt at time t*1


        if torch.isnan(temp_this).any() or torch.isnan(volt_this).any():
            print("Warning: NaN detected in temp_this or volt_this before clamping!")
    
        epsilon = 1e-4
        #temp_this = torch.clamp(temp_this + epsilon, min_temp, max_temp)
        #volt_this = torch.clamp(volt_this + epsilon, min_volt, max_volt)

        #temp_last = torch.clamp(temp_last, min_temp, max_temp)
        #volt_last = torch.clamp(volt_last, min_volt, max_volt)

  

        if torch.isnan(values_this).any() or torch.isnan(values_last).any():
            print("Warning: NaN detected in values_this or values_last after clamping!")
    
        dvdt = torch.abs((temp_this-temp_last)/self.delta_t)
        if torch.isnan(dvdt).any():
            print("Warning: NaN detected in dvdt!")
            
        grad_value = self.gradop(graph, values_this)
        grad_v = grad_value[1]          # Volt Gradient at t+1
        squared_abs_grad_v = torch.sum(grad_v ** 2, dim=1, keepdim=True)  # Shape (N, 1)

        if torch.isnan(squared_abs_grad_v).any():
            print("Warning: NaN detected in squared_abs_grad_v!")
            
        sigma = f*(1+g*(temp_this - e)) # Sigma at t+1
        q = sigma*squared_abs_grad_v    # q at t+1

        lap_value = self.laplacianop(graph,values_this)
    
        lap_temp = lap_value[:,0:1]
        lap_volt = lap_value[:,1:2]

        if torch.isnan(lap_temp).any() or torch.isnan(lap_volt).any():
            print("Warning: NaN detected in lap_temp or lap_volt!")
            
        loss_volt = sigma*lap_volt
        loss_temp = -0.01*(q + c*lap_temp + (d*(e-temp_this)) - (a*b*dvdt)) 
        print("losses_tempthen_volt")
        print(loss_temp)
        print(loss_volt)
        if torch.isnan(loss_temp).any() or torch.isnan(loss_volt).any():
            print("Warning: NaN detected in loss_temp or loss_volt!")

    #    condition = (volt_this >= 6) & (temp_this < 320)
    #    temp_this[condition] = 320


        return torch.cat([loss_temp,loss_volt],axis=1)

    
    

    
    
