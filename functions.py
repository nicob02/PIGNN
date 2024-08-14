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
        graph.x = torch.cat((graph.x,q), dim=-1)    # Append new Q value at t to the input
        return graph    

    def init_condition(self, pos):
        
        temp = torch.full_like(pos[:, 0:1], 310)  # Create a tensor filled with 310s for the temperature
        volt = torch.full_like(pos[:, 1:2], 0)    # Create a tensor filled with 1s for the voltage
    
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
        values_this = (values_this - torch.mean(values_this, dim=0)) / torch.std(values_this, dim=0)
        values_last = (values_last - torch.mean(values_last, dim=0)) / torch.std(values_last, dim=0)
        max_temp = 400  # Maximum realistic temperature
        min_temp = 310  # Minimum realistic temperature (e.g., absolute zero)
        min_volt = 0
        max_volt = 25
        #temp_this = torch.clamp(temp_this, min_temp, max_temp)
        #volt_this = torch.clamp(volt_this, min_volt, max_volt)
        
        temp_last = values_last[:,0:1]  # Temp at time t
        volt_last = values_last[:,1:2]  # Volt at time t
        temp_this = values_this[:,0:1]  # Temp at time t+1
        volt_this = values_this[:,1:2]  # Volt at time t*1
        
        dvdt = torch.abs((temp_this-temp_last)/self.delta_t)
        grad_value = self.gradop(graph, values_this)
        grad_v = grad_value[1]          # Volt Gradient at t+1
        squared_abs_grad_v = torch.sum(grad_v ** 2, dim=1, keepdim=True)  # Shape (N, 1)
        sigma = f*(1+g*(temp_this - e)) # Sigma at t+1
        q = sigma*squared_abs_grad_v    # q at t+1
       
        print("squared_abs_grad_v")
        print(squared_abs_grad_v)
        print("values_this")
        print(values_this)
        print("values_last")
        print(values_last)
        lap_value = self.laplacianop(graph,values_this)

        lap_temp = lap_value[:,0:1]
        lap_volt = lap_value[:,1:2]

 
        #abs_lap_v = torch.sqrt(lap_volt ** 2)

        print("lap_volt")
        print(lap_volt)
        loss_volt = -sigma*lap_volt
        print("loss_volt")
        print(loss_volt)


        
        loss_temp = -(q + (c*lap_temp) + (d*(e-temp_this)) - (a*b*dvdt))

   
        #loss_volt = loss_volt / torch.max(torch.abs(loss_volt))

        #loss_temp = loss_temp / torch.max(torch.abs(loss_temp))
        
        loss_temp = -0.0001*loss_temp
        print("lap_temp")
        print(lap_temp)
        print("loss_temp")
        print(loss_temp)
    
        return torch.cat([loss_temp,loss_volt],axis=1)

    
    

    
    
