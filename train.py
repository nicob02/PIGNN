import torch
from core.models import msgPassing
from core.geometry import ElectrodeMesh
from torch.utils.tensorboard import SummaryWriter
from core.utils.tools import parse_config, modelTrainer
from functions import ElectroThermalFunc as Func
import matplotlib.pyplot as plt

device = torch.device(0)

delta_t = 0.4e-6 # Mess around with this

#func_name = 'rfa'
out_ndim = 2
rfa_params = [1060 , 3600 , 0.512 , 5 , 310 , 0.33 , 0.02 ]  
# Liver density ρti, Liver Heat Capacity dti, liver thermal conductivity dti, Convective
# Transfer coefficient H, Blood/Ground Temp Tbl, Liver Electrical Conductivity σti(T) increases linearly by 2%


ckptpath = 'checkpoint/simulator_%s.pth' % Func.func_name  #Check this out

func_main = Func(delta_t=delta_t, params=rfa_params)

ic = func_main.init_condition
bc1 = func_main.boundary_condition
bc2 = func_main.electrode_condition

model = msgPassing(message_passing_num=1, node_input_size=out_ndim, edge_input_size=3, 
                   ndim=out_ndim, device=device, model_dir=ckptpath)    # Mess with MPN# to 2 or 3
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

mesh = ElectrodeMesh(ru=(0.7, 0.7), lb=(0.3, 0.3), density=35)

graph = mesh.getGraphData().to(device)

print("mesh")

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

print("bc1 and bc2 nodes")

on_boundary = torch.squeeze(graph.node_type == ElectrodeMesh.node_type_ref.boundary)  
on_electrode = torch.squeeze(graph.node_type == ElectrodeMesh.node_type_ref.electrode)  

# Get indices and move them to CPU
electrode_indices = torch.where(on_electrode)[0].cpu()
boundary_indices = torch.where(on_boundary)[0].cpu()

electrode_positions = mesh.pos[electrode_indices.numpy()]
boundary_positions = mesh.pos[boundary_indices.numpy()]

plt.figure(figsize=(8, 8))

# Plot the entire mesh
plt.triplot(mesh.pos[:, 0], mesh.pos[:, 1], mesh.faces.T, color='lightgray')

# Plot the boundary nodes
plt.scatter(boundary_positions[:, 0], boundary_positions[:, 1], color='blue', s=10, label='Boundary Nodes')

# Plot the electrode nodes
plt.scatter(electrode_positions[:, 0], electrode_positions[:, 1], color='red', s=10, label='Electrode Nodes')

plt.title('Boundary and Electrode Nodes')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.savefig('bc1bc2_plot.png')  # Save the figure to a file
plt.show()


    
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
setattr(train_config, 'train_steps', 20)    # 1 minute total simulation
setattr(train_config, 'epchoes', 1500)
setattr(train_config, 'NodeTypesRef', ElectrodeMesh.node_type_ref) 
setattr(train_config, 'step_times', 1)
#setattr(train_config, 'name', func_name)
setattr(train_config, 'ndim', out_ndim)
setattr(train_config, 'lrstep', 100) #learning rate decay epchoes
setattr(train_config, 'writer', writer)
setattr(train_config, 'func_main', func_main)


modelTrainer(train_config)

