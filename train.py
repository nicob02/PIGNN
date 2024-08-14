import torch
from core.models import msgPassing
from core.geometry import ElectrodeMesh
from torch.utils.tensorboard import SummaryWriter
from core.utils.tools import parse_config, modelTrainer
from functions import ElectroThermalFunc as Func
import matplotlib.pyplot as plt

device = torch.device(0)

delta_t = 0.01 # Mess around with this

#func_name = 'rfa'
out_ndim = 2
rfa_params = [1060 , 3600 , 0.512 , 244000 , 310 , 0.33 , 0.02 ]  # Go with Kelvin just in case


ckptpath = 'checkpoint/simulator_%s.pth' % Func.func_name  #Check this out

func_main = Func(delta_t=delta_t, params=rfa_params)

ic = func_main.init_condition
bc1 = func_main.boundary_condition
bc2 = func_main.electrode_condition

model = msgPassing(message_passing_num=1, node_input_size=4+out_ndim, edge_input_size=3, 
                   ndim=out_ndim, device=device, model_dir=ckptpath)    # Mess with MPN# to 2 or 3
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

mesh = ElectrodeMesh(ru=(1, 1), lb=(0, 0), density=100)

print("mesh")



# Assuming your ElectrodeMesh class has attributes like pos (positions of nodes) and faces (connectivity)
mesh = ElectrodeMesh(ru=(1, 1), lb=(0, 0), density=100)

# Extract node positions and connectivity
pos = mesh.pos  # Shape (N, 2), where N is the number of nodes
faces = mesh.faces  # Shape (3, M), where M is the number of triangular elements

# Plot the mesh
plt.figure(figsize=(8, 8))
plt.triplot(pos[:, 0], pos[:, 1], faces.T, color='blue', linewidth=0.5)
plt.scatter(pos[:, 0], pos[:, 1], color='red', s=1)  
plt.title('Mesh Geometry')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('mesh_plot.png')  # Save the figure to a file
plt.show()

graph = mesh.getGraphData().to(device)


    
train_config = parse_config()
writer = SummaryWriter('runs/%s' % Func.func_name)   
 
setattr(train_config, 'pde', func_main.pde)
setattr(train_config, 'graph_modify', func_main.graph_modify)        
setattr(train_config, 'delta_t', delta_t)
setattr(train_config, 'ic', ic)
setattr(train_config, 'bc1', bc1)
setattr(train_config, 'bc2', bc2)
setattr(train_config, 'graph', graph)
setattr(train_config, 'model', model)
setattr(train_config, 'optimizer', optimizer)
setattr(train_config, 'train_steps', 60)    # 1 minute total simulation
setattr(train_config, 'epchoes', 2)
setattr(train_config, 'NodeTypesRef', ElectrodeMesh.node_type_ref) 
setattr(train_config, 'step_times', 1)
#setattr(train_config, 'name', func_name)
setattr(train_config, 'ndim', out_ndim)
setattr(train_config, 'lrstep', 100) #learning rate decay epchoes
setattr(train_config, 'writer', writer)
setattr(train_config, 'func_main', func_main)

modelTrainer(train_config)
    
