from torch_geometric.data import Data
import torch
from collections import defaultdict
from tqdm import tqdm


class SolveWeightLST2d(object):
    '''
    laplacian weights
    '''
    def __init__(self):
        
        def func(pos):

            x = pos[:, 0:1]
            y = pos[:, 1:2]
            v = torch.cat([x, y, x*y, x*x, y*y], dim=-1)
            return v

        def laplacian_func(pos):

            v = torch.zeros((pos.shape[0], 5),
                            dtype=pos.dtype, device=pos.device)
            v[:, 3] = 2     # derivative of y squared is 2y.
            v[:, 4] = 2
            return v

            
        self.func = func
        self.laplacian_func = laplacian_func


    def __call__(self, data:Data):

        pos = data.pos
        edges = data.edge_index

        number_nodes = pos.shape[0]
        weights = torch.zeros_like(edges[1], dtype=torch.float)
        
        lap = self.laplacian_func(pos)
        diff_ = self.func(pos[edges[1]] - pos[edges[0]]) - 0             
        
        all_A_dict = defaultdict(list)
        all_B_dict = defaultdict(list)
        index_dict = defaultdict(list)
        
        for i in tqdm(range(number_nodes)):

            diff = diff_[edges[1]==i]       # Basis function of the difference between solutions of nodes on edge. u(xi - xj)
            laplacian_value = lap[i:i+1]

            A = diff.t()
            neibor = A.shape[1]
            B = laplacian_value.t() # Pre-computed laplacian value, triangle(u)
            
            all_A_dict[neibor].append(A)
            all_B_dict[neibor].append(B)
            index_dict[neibor].append(i)
        
        for n in all_A_dict.keys():   # Loops through each key, i.e. all nodes having the same number of neighbours.
            A = torch.stack(all_A_dict[n], dim=0)
            B = torch.stack(all_B_dict[n], dim=0)
            index = index_dict[n]   # Get the indices of all nodes with that #neighb
            X = torch.linalg.lstsq(A, B).solution   # Gets the weights
            for i, w in enumerate(X):   # i is current index of iteration with #nodeswith(n)neighbours total iterations, w is its weights
                receiver = index[i]     # this will give u the actual target node index
                w = w.squeeze()
                weights[edges[1]==receiver] = w #receiver as that is the one we care about based on neighbours logic thing from before.
                    
        weights = weights.detach()
        return weights



    
    


