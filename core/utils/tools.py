import json
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil

def RemoveDir(filepath):
    '''
    If the folder doesn't exist, create it; and if it exists, clear it.
    '''
    if not os.path.exists(filepath):
        os.makedirs(filepath,exist_ok=True)
    else:
        shutil.rmtree(filepath)
        os.makedirs(filepath, exist_ok=True)


class Config:
    def __init__(self) -> None:
        pass
    def __setattr__(self, __name: str, __value) -> None:
        self.__dict__[__name] = __value


def parse_config(file='config.json'):
    configs = Config() 
    if not os.path.exists(file):
        return configs
    with open(file, 'r') as f:
        data = json.load(f)
        for k, v in data.items():
            config = Config()
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    config.setattr(k1, v1)
            else:
                raise TypeError
            configs[k] = config
    return configs[k]


def modelTrainer(config):
    
    delta_t = config.delta_t
    model = config.model
    graph = config.graph
    #scheduler = torch.optim.lr_scheduler.StepLR(
        #config.optimizer, step_size=config.lrstep, gamma=0.99)  
    config.optimizer.zero_grad()
    
    best_loss  = np.inf
    
    for epcho in range(1, config.epchoes + 1):  # Creates different ic and solves the problem, does this epoch # of times

        graph.x = config.ic(graph.pos)
        
        begin_time = 0
        total_steps_loss = 0
        on_boundary = torch.squeeze(graph.node_type == config.NodeTypesRef.boundary)  
        on_electrode = torch.squeeze(graph.node_type == config.NodeTypesRef.electrode)  
        #has_electrode = torch.any(on_electrode)
        #print(f"on_electrode has any True values: {has_electrode}")

        config.optimizer.zero_grad()
            
        losses = {}
        for step in range(1, config.train_steps + 1):      # Goes through the whole simulation for that epoch   
        
            this_time = begin_time + delta_t * step            
            #graph.x[on_boundary] = boundary_value[on_boundary]
            value_last = graph.x.detach().clone()
            
            config.graph_modify(graph, value_last=value_last)            
            predicted = model(graph)   
            # hard boundary         
            boundary_value = config.bc1(graph.pos)
            predicted[on_boundary] = boundary_value[on_boundary] 
            electrode_value = config.bc2(graph.pos, predicted, this_time)
            predicted[on_electrode] = electrode_value[on_electrode]

            pde_loss = config.pde(graph, values_last=value_last, values_this=predicted)
            pde_loss[on_boundary] = 0
            pde_loss[on_electrode] = 0
            loss = torch.norm(pde_loss)/pde_loss.numel()
                
            loss.backward()
            graph.x = predicted.detach()

            losses.update({"step%d" % step: loss.detach()})
            total_steps_loss += loss.item()/config.train_steps
            
        config.writer.add_scalars("loss", losses, epcho)
        config.writer.add_scalar("total_steps_loss", total_steps_loss, epcho)
        config.writer.flush()
        config.optimizer.step()       # Updates the state's model paramaters
        
        if total_steps_loss < best_loss:
            best_loss  = total_steps_loss
            model.save_model(config.optimizer)
            print('model saved at loss: %.4e' % best_loss) 
            
        #scheduler.step()       
        
    print('Training completed! Model saved to %s'%config.model.model_dir)
        
@torch.no_grad()            # Disables gradient computations for evaluation
def modelTester(config):
    
    delta_t = config.delta_t
    model = config.model.to(config.device)
    config.graph = config.graph.to(config.device)


    test_steps = config.test_steps   
    config.graph.x = config.ic(config.graph.pos)    
    
    begin_time = 0
    test_results = []
    on_boundary = torch.squeeze(config.graph.node_type==config.NodeTypesRef.boundary)
    on_electrode = torch.squeeze(config.graph.node_type == config.NodeTypesRef.electrode)
    boundary_value = config.bc1(config.graph.pos)     

    def predictor(model, graph, step):
        this_time = begin_time + delta_t * step
        value_last = graph.x.detach().clone()
        graph.x[on_boundary] = boundary_value[on_boundary]
        config.graph_modify(config.graph, value_last=value_last)
        predicted = model(graph)
        electrode_value = config.bc2(graph.pos, predicted, this_time)
        predicted[on_boundary] = boundary_value[on_boundary]
        predicted[on_electrode] = electrode_value[on_electrode]

        return predicted

    for step in tqdm(range(1, test_steps + 1)):      
        v = predictor(model, config.graph, step)
        config.graph.x = v.detach()
        v = v.clone().cpu().numpy()        
        test_results.append(v)    
    
    v = np.stack(test_results, axis=0)   
    return v


# Averages MSQR error over all nodes for each time step, then cumsum of all time steps normalized by #timesteps
def rollout_error_test(predicteds, targets):
    number_len = targets.shape[0] 
    squared_diff = np.square(predicteds - targets).reshape(number_len, -1)   
    loss = np.sqrt(np.cumsum(np.mean(squared_diff, axis=1), 
                             axis=0)/np.arange(1, number_len + 1))  
    return loss


def render_results(predicteds, reals, config, save_dir):

    if predicteds is None: return
    test_steps = config.test_steps   
    pos = config.graph.pos.cpu().numpy()
    x = pos[:, 0]
    y = pos[:, 1] 

    diffs = np.abs(predicteds - reals)
    # diffs = np.abs((predicteds - reals)/reals)    # Relative error not used at the end

    real_max  = np.max(reals[:, :, 0])  # Max temp value
    real_min = np.min(reals[:, :, 0])   # Min temp value, would need a 1 instead of 0 to get volt

    diff_max = np.max(diffs[:, :, 0])   # Since I'm only interested in temp dist. and not volt this is good
    diff_min = np.min(diffs[:, :, 0]) 

    for index_ in tqdm(range(1, test_steps, 9)):
     
        predicted = predicteds[index_]   # Model prediction at time step index_
        real = reals[index_]
        diff = diffs[index_]        

        data_index = 0      # Indicates component 0(temp) is being visualized
        fig, axes = plt.subplots(1, 3, figsize=(16, 5)) 

        for idx, ax in enumerate(axes):    

            if idx == 0:
                s_r = ax.scatter(x, y, c=real[:, data_index], alpha=0.95, cmap='seismic', \
                    marker='s', s=5, vmin=real_min, vmax=real_max)
                ax.set_title('Exact @ step: %d'%(index_+1),fontsize=10)  
                plt.colorbar(s_r, ax=ax) 
            elif idx == 1:
                s_p = ax.scatter(x, y, c=predicted[:, data_index], alpha=0.95, cmap='seismic',\
                     marker='s', s=5, vmin=real_min, vmax=real_max) 
                ax.set_title('Predicted @ step: %d'%(index_+1),fontsize=10)
                plt.colorbar(s_p, ax=ax) 
            elif idx == 2: 
                s_d = ax.scatter(x, y, c=diff[:, data_index], alpha=0.95, cmap='seismic', \
                    marker='s', s=5, vmin=diff_min, vmax=diff_max)                
                ax.set_title('Difference @ step: %d'%(index_+1),fontsize=10)   
                plt.colorbar(s_d,ax=ax)

        # fig.colorbar(s_r, ax=axes). No shared color bar but unique one for all 3 subplots
        # Creates a file with the temp dist plots for each time step
        plt.savefig(save_dir+'testResults_step_%d.png'%(index_+1), bbox_inches = 'tight')   
        plt.close()
        
def render_temperature(predicteds, graph):
    test_begin_step = 0
    if predicteds is None:
        return
    total_test_steps = predicteds.shape[0]
    pos = graph.pos.cpu().numpy()

    temp_max = np.max(predicteds[:, :, 0])
    temp_min = np.min(predicteds[:, :, 0])

    os.makedirs('images', exist_ok=True)
    x = pos[:, 0]
    y = pos[:, 1]

    for index_ in tqdm(range(total_test_steps)):
        if index_ % 20 != 0:
            continue
        predicted = predicteds[index_]

        data_index = 0  #index representing temp

        c = predicted[:, data_index:data_index+1]

        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        b = axes.scatter(x, y, c=c,  vmin=temp_min, vmax=temp_max, cmap="plasma")
        fig.colorbar(b, ax=axes)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.savefig('images/result%dtemp_predict.png' %
                    (test_begin_step+index_), bbox_inches='tight')
        plt.close()


        data_index = 1  #volt index

        v_max = np.max(predicted[:,  data_index])
        v_min = np.min(predicted[:,  data_index])

        c = predicted[:, data_index:data_index+1]

        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        b = axes.scatter(x, y, c=c,  vmin=v_min, vmax=v_max, cmap="plasma")
        fig.colorbar(b, ax=axes)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.savefig('images/result%dv_predict.png' %
                    (test_begin_step+index_), bbox_inches='tight')
        plt.close()


def plot_error_curve(error, begin_step, config, save_dir):

    delta_t = config.delta_t
    number_len = error.shape[0]
    fig, axes = plt.subplots(1, 1, figsize=(8,5))
    axes.set_yscale("log")
    axes.plot((begin_step + np.arange(number_len)) * delta_t, error)
    axes.set_xlim(begin_step * delta_t, (begin_step + number_len) * delta_t)
    axes.set_ylim(5e-5, 10) #Mess around with this if my error differs more than expected
    axes.set_xlabel('time (s)')
    axes.set_ylabel('RMSE')
    
    my_x1 = np.linspace(begin_step * delta_t, (begin_step + number_len - 1) * delta_t, 25)  #Check later whether 25 are too many values or too little
    plt.xticks(my_x1)
    plt.title('Error Curve')
    plt.savefig(save_dir + '%s_rollout_aRMSE_Parameters[%d]_dens[%d]_Steps[%d].png'%(config.name, \
        config.rfa_params, config.density, config.test_steps))
    plt.close()       
        

        

    
    
    
