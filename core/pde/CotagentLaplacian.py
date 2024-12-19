import torch
from torch_geometric.data import Data
from torch_scatter import scatter_add
import math

class CotangentLaplacian:
    def __init__(self):
        pass

    def __call__(self, data: Data):
        """
        Construct the cotangent Laplacian matrix.
        data should contain:
          - data.pos: (N, 2) node coordinates
          - data.faces: (F, 3) indices of triangular faces (long tensor)
        Returns a torch sparse tensor representing the Laplacian matrix L.
        """
        if not hasattr(data, 'faces'):
            raise ValueError("data must have 'faces' attribute for cotangent Laplacian.")

        pos = data.pos
        faces = data.faces  # faces shape: (F, 3)

        # Compute cotangent weights
        # For each face, we have three edges and three angles.
        # The cotan weight for an edge is computed by looking at the angles opposite that edge in the adjacent faces.

        # Step 1: Compute all edges of each face and angles
        # For a triangle with vertices (a,b,c), the angles opposite edges are computed by vector operations.

        # We'll create a helper function:
        def compute_cotan_angles(pos, faces):
            # pos: (N,2), faces: (F,3)
            # returns per-face angles and edges
            a = pos[faces[:, 0]]
            b = pos[faces[:, 1]]
            c = pos[faces[:, 2]]

            # Edges vectors
            ab = b - a
            bc = c - b
            ca = a - c

            # Lengths
            ab_len = (ab**2).sum(dim=1).sqrt()
            bc_len = (bc**2).sum(dim=1).sqrt()
            ca_len = (ca**2).sum(dim=1).sqrt()

            # Angles: 
            # For angle at 'a' opposite edge BC:
            # cos(alpha_a) = (|ab|^2 + |ac|^2 - |bc|^2) / (2|ab||ac|)
            # But we want cot(alpha), so let's use vector dot product to find angles:
            # cot(alpha) = cos(alpha)/sin(alpha)
            # A robust approach: For angle at A formed by vectors (a->b) and (a->c):
            # alpha_A = angle between (b - a) and (c - a)
            # cot(alpha_A) = ( (b-a)·(c-a) ) / cross((b-a),(c-a))
            
            # Cross product magnitude in 2D: cross(u,v) = ux*vy - uy*vx
            # We'll do each angle:
            # angle at A opposite BC:
            v1 = b - a
            v2 = c - a
            cross_a = v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]
            dot_a = (v1 * v2).sum(dim=1)
            cot_a = dot_a / (cross_a + 1e-14)

            # angle at B opposite CA:
            v1 = a - b
            v2 = c - b
            cross_b = v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]
            dot_b = (v1 * v2).sum(dim=1)
            cot_b = dot_b / (cross_b + 1e-14)

            # angle at C opposite AB:
            v1 = a - c
            v2 = b - c
            cross_c = v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]
            dot_c = (v1 * v2).sum(dim=1)
            cot_c = dot_c / (cross_c + 1e-14)

            return cot_a, cot_b, cot_c

        cot_a, cot_b, cot_c = compute_cotan_angles(pos, faces)

        # cot_a corresponds to angle at A, opposite edge BC
        # For edge opposite A: edge (b,c)
        # For edge opposite B: edge (a,c)
        # For edge opposite C: edge (a,b)

        # We'll now build the sparse Laplacian:
        # Edges from faces:
        # Opposite A: edge (faces[:,1], faces[:,2]) -> weight contribution cot_a
        # Opposite B: edge (faces[:,0], faces[:,2]) -> weight contribution cot_b
        # Opposite C: edge (faces[:,0], faces[:,1]) -> weight contribution cot_c

        # Each edge will get contributions from the two adjacent faces. We'll sum them.
        i0, i1, i2 = faces[:,0], faces[:,1], faces[:,2]

        # We'll create triplets (row, col, value) for each half-edge and sum them up.
        # For edge (b,c) add cot_a/2 to w_bc and w_cb
        # Actually, for a closed surface, w_ij = (cot(alpha)+cot(beta))/2
        # Here each face gives us a cot value for an edge. If we have two faces sharing the edge, both contribute.
        
        # Initialize lists
        row = []
        col = []
        val = []

        def add_edge_contribution(i1, i2, c):
            # Add symmetric contributions
            row.append(i1)
            col.append(i2)
            val.append(c/2)
            row.append(i2)
            col.append(i1)
            val.append(c/2)

        # edges (i1,i2) opposite A
        add_edge_contribution(i1, i2, cot_a)
        # edges (i0,i2) opposite B
        add_edge_contribution(i0, i2, cot_b)
        # edges (i0,i1) opposite C
        add_edge_contribution(i0, i1, cot_c)

        row = torch.cat([r if isinstance(r, torch.Tensor) else torch.tensor(r, device=pos.device) for r in row])
        col = torch.cat([c if isinstance(c, torch.Tensor) else torch.tensor(c, device=pos.device) for c in col])
        val = torch.cat([v if isinstance(v, torch.Tensor) else torch.tensor(v, device=pos.device, dtype=pos.dtype) for v in val])

        # Now sum duplicate edges:
        # scatter_add to aggregate duplicates
        # We'll create a key to identify edges uniquely. For undirected edges, sort indices.
        mask = row > col
        r_ = torch.where(mask, col, row)
        c_ = torch.where(mask, row, col)
        # Sort by (r_, c_)
        # We can use a coalesce operation from torch.sparse, or just build a single key:
        key = r_.to(torch.int64) * pos.shape[0] + c_.to(torch.int64)
        sorted_key, perm = key.sort()
        r_ = r_[perm]
        c_ = c_[perm]
        val = val[perm]
        # Sum duplicates
        uniq_key, inv = torch.unique(sorted_key, return_inverse=True)
        val = scatter_add(val, inv)
        r_ = scatter_add(r_.float(), inv, dim=0) # This is just to track them, we only need one instance
        c_ = scatter_add(c_.float(), inv, dim=0) 
        # After scatter, r_ and c_ are summed. But since each group was identical, each group had identical r_, c_ pairs repeated (like a stable average).
        # Let's just re-derive them from uniq_key:
        r_final = (uniq_key // pos.shape[0]).to(torch.long)
        c_final = (uniq_key % pos.shape[0]).to(torch.long)

        # Build adjacency from these cotan weights
        # Construct W matrix
        # Diagonal entries: sum of row weights
        # L = D - W
        # W[i,j] = val for edge (i,j)
        # D[i,i] = sum_j W[i,j]

        # We'll build W as a sparse matrix
        indices = torch.stack([r_final, c_final], dim=0)
        W = torch.sparse_coo_tensor(indices, val, (pos.shape[0], pos.shape[0]))

        # Compute diagonal
        row_sum = scatter_add(val, r_final, dim=0, dim_size=pos.shape[0])
        D = torch.sparse_coo_tensor(torch.stack([torch.arange(pos.shape[0], device=pos.device),
                                                 torch.arange(pos.shape[0], device=pos.device)], dim=0),
                                    row_sum, (pos.shape[0], pos.shape[0]))

        L = D.coalesce() - W.coalesce()
        return L.coalesce()

# Example usage:
# data = Data(pos=pos, faces=faces) # faces: (F,3)
# laplacian = CotangentLaplacian()
# L = laplacian(data) # returns a sparse Laplacian torch tensor

