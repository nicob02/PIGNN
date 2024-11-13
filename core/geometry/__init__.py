from fenics import Point
from mshr import generate_mesh, Rectangle
import numpy as np
from enum import IntEnum
import torch_geometric.transforms as T
from torch_geometric.data import Data
import torch
from dolfin import *

class NodeType(IntEnum):
    inner=0
    boundary=1
    electrode=2

def get_node_type(pos, lb_electrode, ru_electrode, radius_ratio=None):
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 1])
    min_x = np.min(pos[:, 0])
    min_y = np.min(pos[:, 1])
    
    right = np.isclose(pos[:, 0], max_x)
    left = np.isclose(pos[:, 0], min_x)
    up = np.isclose(pos[:, 1], max_y)
    bottom = np.isclose(pos[:, 1], min_y)    
    
    on_boundary = np.logical_or(np.logical_or(right, left),np.logical_or(up, bottom))
    
    
    right_electrode = np.logical_and(np.isclose(pos[:, 0], ru_electrode[0]), np.logical_and(pos[:, 1] >= lb_electrode[1], pos[:, 1] <= ru_electrode[1]))
    left_electrode = np.logical_and(np.isclose(pos[:, 0], lb_electrode[0]), np.logical_and(pos[:, 1] >= lb_electrode[1], pos[:, 1] <= ru_electrode[1]))
    up_electrode = np.logical_and(np.isclose(pos[:, 1], ru_electrode[1]), np.logical_and(pos[:, 0] >= lb_electrode[0], pos[:, 0] <= ru_electrode[0]))
    bottom_electrode = np.logical_and(np.isclose(pos[:, 1], lb_electrode[1]), np.logical_and(pos[:, 0] >= lb_electrode[0], pos[:, 0] <= ru_electrode[0]))

    on_electrode = np.logical_or(np.logical_or(right_electrode, left_electrode), np.logical_or(up_electrode, bottom_electrode))

    node_type = np.ones((pos.shape[0], 1))
    node_type[on_boundary] = NodeType.boundary
    node_type[on_electrode] = NodeType.electrode
    node_type[np.logical_not(np.logical_or(on_boundary, on_electrode))] = NodeType.inner
        
    return np.squeeze(node_type)
    

class ElectrodeMesh():
    
    node_type_ref = NodeType
    def __init__(self, density=30, lb=(0, 0), ru=(1, 1)) -> None:
        
        self.transform = T.Compose([
            T.FaceToEdge(remove_faces=False), 
            T.Cartesian(norm=False), 
            T.Distance(norm=False)
            ])
        #random_center_electrode_x = np.random.uniform(0.05,0.95)   #Electrode probe is placed randomly at each training iteration
        #random_center_electrode_y = np.random.uniform(0.1,0.9)
        
        lb_electrode = [(0.48),(0.45)]
        ru_electrode = [(0.52),(0.55)]
        domain = Rectangle(Point(lb[0],lb[1]), Point(ru[0], ru[1]))  # Geometry Domain
        electrode_probe = Rectangle(Point(lb_electrode[0], lb_electrode[1]), Point(ru_electrode[0], ru_electrode[1]))
        geometry = domain - electrode_probe
        initial_mesh = generate_mesh(geometry, density)
        boundary_markers = MeshFunction("size_t", initial_mesh, initial_mesh.topology().dim() - 1, 0)
        for facet in facets(initial_mesh):
            if facet.midpoint().distance(Point(0.5, 0.5)) < 0.15:
                boundary_markers[facet] = 1  # Mark region near electrode
        #    if facet.midpoint().distance(Point(ru_electrode[0], ru_electrode[1])) < 0.1:
        #       boundary_markers[facet] = 1  # Mark region near electrode
        
        # Refine mesh selectively around electrode
        #for i in range(3):  # Number of refinements
        cell_markers = MeshFunction("bool", initial_mesh, initial_mesh.topology().dim())
        cell_markers.set_all(False)
        for cell in cells(initial_mesh):
            for facet in facets(cell):
                if boundary_markers[facet] == 1:
                    cell_markers[cell] = True  # Mark cells for refinement around electrode
    
        # Refine the mesh around marked cells
        initial_mesh = refine(initial_mesh, cell_markers)
        self.mesh = initial_mesh
        self.pos = self.mesh.coordinates().astype(np.float32)
        self.faces = self.mesh.cells().astype(np.int64).T        
        self.node_type = get_node_type(self.pos, lb_electrode, ru_electrode).astype(np.int64)
        print("Node numbers: %d"%self.pos.shape[0])
        
    def getGraphData(self):
        graph = Data(pos=torch.as_tensor(self.pos), 
                    face=torch.as_tensor(self.faces))
        graph = self.transform(graph)
        graph.num_nodes = graph.pos.shape[0]
        graph.node_type = torch.as_tensor(self.node_type)
        graph.label = 0
        return graph

