import torch
from core.utils.tools import parse_config, modelTester, RemoveDir
from core.utils.tools import rollout_error_test, plot_error_curve, render_results, render_temperature
from core.models import msgPassing
from core.geometry import ElectrodeMesh
from functions import ElectroThermalFunc as Func
import os




delta_t = 1 # Mess around with this
poisson_params = 6.28318

#func_name = 'rfa'
out_ndim = 1


ckptpath = 'checkpoint/simulator_%s.pth' % Func.func_name    #FIGURE THIS OUT
device = torch.device(0)

func_main = Func(delta_t=delta_t, params=poisson_params)

bc1 = func_main.boundary_condition
ic = func_main.init_condition

mesh = ElectrodeMesh(ru=(1, 1), lb=(0, 0), density=65)
graph = mesh.getGraphData()
model = msgPassing(message_passing_num=1, node_input_size=out_ndim+3, 
                   edge_input_size=3, ndim=out_ndim, device=device, model_dir=ckptpath)
model.load_model(ckptpath)
model.to(device)
model.eval()
test_steps = 20

test_config = parse_config()

#model = kwargs['model'] # Extracts the model's dictioanry with the weights and biases values
setattr(test_config, 'rfa_params', rfa_params)
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

#real_results = []
# Can also later measure the time taken for this for loop to be completed vs the model predict_results time taken
#for step in range(1, test_config.test_steps +1):
    #t = step * delta_t
    #v1 = func_main.exact_solution(graph.pos, t)     # This I will surely have to modify the arguments
    #real_results.append(v1)
#real_results = torch.stack(real_results, dim=0).cpu().numpy()


#aRMSE = rollout_error_test(predict_results, real_results) 


#-----------------plotting----------------------------

#results_root = 'NMGNN_%s_Results/'%(test_config.name)

#aRMSE_Fig_save_dir = results_root + 'aRMSE_Fig/'
#os.makedirs(aRMSE_Fig_save_dir, exist_ok = True)
#print('NMGNN_%s_Parameters[%d]_dens[%d]_Steps[%d]: [loss_mean: %.4e]'%(
    #test_config.name, test_config.rfa_params, test_config.density, test_config.test_steps, aRMSE[-1]))  # -1 is last value of array containing average error
#plot_error_curve(aRMSE, 0, test_config, aRMSE_Fig_save_dir)



#testImg_save_dir = results_root + 'testImages_%s_Parameters[%d]_area%s_dens[%d]_Steps[%d]_ALL/'%(\
    #test_config.name, test_config.rfa_params, test_config.density, test_config.test_steps)
#RemoveDir(testImg_save_dir)

render_temperature(predict_results, graph)

#render_results(predict_results, real_results, test_config, testImg_save_dir)
