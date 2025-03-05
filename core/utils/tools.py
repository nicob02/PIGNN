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
    scheduler = torch.optim.lr_scheduler.StepLR(
        config.optimizer, step_size=config.lrstep, gamma=0.99)  
    
    best_loss  = np.inf
    
    for epcho in range(1, config.epchoes + 1):  # Creates different ic and solves the problem, does this epoch # of times

        graph.x = config.ic(graph.pos)
 
        begin_time = 0
        total_steps_loss = 0
        on_boundary = torch.squeeze(graph.node_type == config.NodeTypesRef.boundary)  
        config.optimizer.zero_grad()
        config.graph_modify(config.graph, value_last=value_last)
        losses = {}
        for step in range(1, config.train_steps + 1):      # Goes through the whole simulation for that epoch   

            
            this_time = begin_time + delta_t * step            
            
            #value_last = graph.x.detach().clone()
            #graph.x = config.bc1(config.graph, predicted = value_last)
            
            
            
            predicted = model(graph)
           
            # hard enforced boundary Ansatz
            predicted = config.bc1(config.graph, predicted = predicted)
        
            #predicted[on_boundary] = boundary_value[on_boundary] 

            loss = config.pde(graph, values_this=predicted)

            #loss[on_boundary] = 0        # TAKE THE HARD-ENFORCED OUT LATER TO COMPARE DIFFERENCE
         
            # Aggregate the loss components
            #loss = torch.norm(loss)/loss.numel()
            loss = torch.norm(loss)
    
                
            loss.backward()
            #graph.x = predicted.detach()

        config.optimizer.step()

            
            #losses.update({"step%d" % step: loss.detach()})
            #total_steps_loss += loss.item()/config.train_steps
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
        #config.writer.add_scalars("loss", losses, epcho)
        #config.writer.add_scalar("total_steps_loss", total_steps_loss, epcho)
        #config.writer.flush()
        #config.optimizer.step()       # Updates the state's model paramaters
        
        if total_steps_loss < best_loss:
            best_loss  = total_steps_loss
            model.save_model(config.optimizer)
            print('model saved at loss: %.4e' % best_loss) 
            
        scheduler.step()       
        
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
      

    def predictor(model, graph, step):
        this_time = begin_time + delta_t * step
        value_last = graph.x.detach().clone()
        #graph.x = config.bc1(config.graph, predicted = value_last)
        config.graph_modify(config.graph, value_last=value_last)
        predicted = model(graph)
        predicted = config.bc1(config.graph, predicted = predicted)
        #boundary_value = config.bc1(config.graph, predicted = predicted) 
        #predicted[on_boundary] = boundary_value[on_boundary]

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


import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def render_results(predicteds, reals, graph):
    if predicteds is None:
        return
    
    # Extract node coordinates from the graph
    pos = graph.pos.cpu().numpy()
    x = pos[:, 0]
    y = pos[:, 1]
    
    os.makedirs('images2', exist_ok=True)
    
    # Compute absolute differences
    diffs = np.abs(predicteds - reals)
    
    # Set up range for 'Exact' and 'Predicted'
    # (Assuming your data is physically within [0,1] for Volts)
    vmin_val, vmax_val = 0.0, 1.0

    # Differences might exceed 0‒1, so we derive from the data
    diff_max = np.max(diffs[:, :, 0])
    diff_min = np.min(diffs[:, :, 0])

    # Render results for the first 5 steps (adjust if needed)
    for index_ in tqdm(range(5)):
        predicted = predicteds[index_]
        real = reals[index_]
        diff = diffs[index_]

        data_index = 0  # Visualizing the 0th component (e.g., temperature/voltage)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for idx, ax in enumerate(axes):
            # Label the axes for each subplot
            ax.set_xlabel('x(m)')
            ax.set_ylabel('y(m)')

            if idx == 0:
                # Exact
                s_r = ax.scatter(
                    x, y, c=real[:, data_index], alpha=0.95, cmap='seismic',
                    marker='s', s=5, vmin=vmin_val, vmax=vmax_val
                )
                ax.set_title('Exact', fontsize=10)
                cb = plt.colorbar(s_r, ax=ax)
                cb.set_label('', labelpad=-30, rotation=0, fontsize=10, loc='top')
            elif idx == 1:
                # Predicted
                s_p = ax.scatter(
                    x, y, c=predicted[:, data_index], alpha=0.95, cmap='seismic',
                    marker='s', s=5, vmin=vmin_val, vmax=vmax_val
                )
                ax.set_title('Predicted', fontsize=10)
                cb = plt.colorbar(s_p, ax=ax)
                cb.set_label(' ', labelpad=-30, rotation=0, fontsize=10, loc='top')
            else:
                # Difference
                s_d = ax.scatter(
                    x, y, c=diff[:, data_index], alpha=0.95, cmap='seismic',
                    marker='s', s=5, vmin=diff_min, vmax=diff_max
                )
                ax.set_title('Difference', fontsize=10)
                cb = plt.colorbar(s_d, ax=ax)
                cb.set_label('', labelpad=-30, rotation=0, fontsize=10, loc='top')

        # Save each figure to file
        plt.savefig(f'images2/result{index_+1}.png', bbox_inches='tight')
        plt.close()

        
def render_temperature(predicteds, graph):
    test_begin_step = 0
    if predicteds is None:
        return
    total_test_steps = predicteds.shape[0]
    pos = graph.pos.cpu().numpy()

    os.makedirs('images', exist_ok=True)
    x = pos[:, 0]
    y = pos[:, 1]

    for index_ in tqdm(range(total_test_steps)):
        #if index_ % 3 != 0:
        #   continue
        predicted = predicteds[index_]
        
        data_index = 0  #volt index

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
    axes.set_ylim(5e-5, 10)
    axes.set_xlabel('time (s)')
    axes.set_ylabel('RMSE')
    
    my_x1 = np.linspace(begin_step * delta_t, (begin_step + number_len - 1) * delta_t, 11)
    plt.xticks(my_x1)
    plt.title('Error Curve')
    plt.savefig(save_dir + '%s_rollout_aRMSE_Reynold[%d]_area%s_dens[%d]_Steps[%d].png'%(config.name, \
        config.Reynold, config.area, config.density, config.test_steps))
    plt.close()       

        

        

    
    
    

