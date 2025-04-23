import torch
from core.utils.tools import parse_config, modelTester, RemoveDir
from core.utils.tools import rollout_error_test, plot_error_curve, render_results, render_temperature
from core.models import msgPassing
from core.geometry import ElectrodeMesh
from functions import ElectroThermalFunc as Func
import os




delta_t = 1 # Mess around with this
#poisson_params = 6.28318
#poisson_params = 12.56637  #4pi, initial test case, change later.
poisson_params = 25.13274  #8pi, initial test case, change later.
#func_name = 'rfa'
out_ndim = 1

dens=65
ckptpath = 'checkpoint/simulator_%s.pth' % Func.func_name    #FIGURE THIS OUT
device = torch.device(0)

func_main = Func(delta_t=delta_t, params=poisson_params)

bc1 = func_main.boundary_condition
ic = func_main.init_condition

mesh = ElectrodeMesh(ru=(1, 1), lb=(0, 0), density=65)
graph = mesh.getGraphData()
model = msgPassing(message_passing_num=2, node_input_size=out_ndim+3, 
                   edge_input_size=3, ndim=out_ndim, device=device, model_dir=ckptpath)
model.load_model(ckptpath)
model.to(device)
model.eval()
test_steps = 20

test_config = parse_config()

#model = kwargs['model'] # Extracts the model's dictioanry with the weights and biases values
setattr(test_config, 'poisson_params', poisson_params)
setattr(test_config, 'delta_t', delta_t)
setattr(test_config, 'device', device)   
setattr(test_config, 'ic', ic)
setattr(test_config, 'bc1', bc1)
setattr(test_config, 'model', model)
setattr(test_config, 'test_steps', test_steps)
setattr(test_config, 'NodeTypesRef', ElectrodeMesh.node_type_ref)
#setattr(test_config, 'name', func_name)
setattr(test_config, 'ndim', out_ndim)
setattr(test_config, 'graph_modify', func_main.graph_modify)
setattr(test_config, 'graph', graph)
setattr(test_config, 'density', dens)

      

#-----------------------------------------

print('************* model test starts! ***********************')
predict_results = modelTester(test_config)

real_results = []
# Can also later measure the time taken for this for loop to be completed vs the model predict_results time taken
for step in range(1, test_config.test_steps +1):
    #t = step * delta_t
    v1 = func_main.exact_solution(graph)     # This I will surely have to modify the arguments
    real_results.append(v1)
real_results = torch.stack(real_results, dim=0).cpu().numpy()


aRMSE = rollout_error_test(predict_results, real_results) 
print("aRMSE")
print(aRMSE)

#-----------------plotting----------------------------


render_temperature(predict_results, graph)
#render_temperature(real_results, graph)
render_results(predict_results, real_results, graph)
