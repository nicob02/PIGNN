from torch_geometric.data import Data
import torch
from collections import defaultdict
from tqdm import tqdm

class SolveWeightLST2d(object):
    '''
    Laplacian weights calculation via a local least-squares approach.

    Changes made:
    - Removed the cross term x*y from the basis to reduce potential ill-conditioning.
    - Added condition number checks for each local system.
    - Added a small Tikhonov regularization term if the condition number is too high.
    '''

    def __init__(self, regularization=1e-10, cond_warning=1e7):
        """
        Parameters:
            regularization: float
                A small Tikhonov regularization parameter to add to AᵀA if needed.
            cond_warning: float
                Condition number threshold above which a warning is printed.
        """
        self.regularization = regularization
        self.cond_warning = cond_warning

        def func(pos):
            # New simpler polynomial basis: [x, y, x², y²]
            x = pos[:, 0:1]
            y = pos[:, 1:2]
            # v: [x, y, x², y²]
            v = torch.cat([x, y, x*x, y*y], dim=-1)
            return v

        def laplacian_func(pos):
            # The Laplacian of [x, y, x², y²]:
            # ∆x = 0
            # ∆y = 0
            # ∆(x²) = 2
            # ∆(y²) = 2
            # So we have 4 basis functions, the last two correspond to second order terms.
            v = torch.zeros((pos.shape[0], 4), dtype=pos.dtype, device=pos.device)
            v[:, 2] = 2.0
            v[:, 3] = 2.0
            return v

        self.func = func
        self.laplacian_func = laplacian_func

    def __call__(self, data: Data):
        pos = data.pos
        edges = data.edge_index

        number_nodes = pos.shape[0]
        weights = torch.zeros_like(edges[1], dtype=torch.float)

        lap = self.laplacian_func(pos)
        diff_ = self.func(pos[edges[1]] - pos[edges[0]])  # Removed '-0' (no effect)

        all_A_dict = defaultdict(list)
        all_B_dict = defaultdict(list)
        index_dict = defaultdict(list)

        # Build local systems
        for i in tqdm(range(number_nodes)):
            diff = diff_[edges[1] == i]       # shape [#neighbors, #basis=4]
            laplacian_value = lap[i:i+1]      # shape [1,4]

            A = diff.t()  # A: [4, #neighbors]
            B = laplacian_value.t()  # B: [4,1]

            neibor = A.shape[1]

            # Condition number check:
            # Compute condition number of A using SVD
            # If #neighbors < 4 (less equations than unknowns), we can't solve stablely anyway
            if neibor < 4:
                # Not enough neighbors - consider skipping or adding neighbors
                # We'll skip this node (no update to weights)
                print(f">4 Neighbours for node")
                continue

            try:
                U, S, Vt = torch.linalg.svd(A, full_matrices=False)
                cond_number = (S[0] / (S[-1] + 1e-14)).item()
                if cond_number > self.cond_warning:
                    print(f"Warning: High condition number {cond_number} at node {i} with {neibor} neighbors.")
            except RuntimeError as e:
                print(f"SVD failed for node {i}: {e}")
                continue

            all_A_dict[neibor].append((A, B, i))

        # Solve local systems by group of same neighbor count
        for n in all_A_dict.keys():
            # Stack all A and B
            A_list = [item[0] for item in all_A_dict[n]]
            B_list = [item[1] for item in all_A_dict[n]]
            index_list = [item[2] for item in all_A_dict[n]]

            A_block = torch.stack(A_list, dim=0) # shape [batch, 4, n]
            B_block = torch.stack(B_list, dim=0) # shape [batch, 4, 1]

            # Solve with least squares: 
            # Normally: X = A⁺ B = (AᵀA)⁻¹AᵀB
            # Add Tikhonov regularization if needed:
            # Instead of directly calling lstsq, we can do:
            # (AᵀA + λI)x = AᵀB
            # We'll try lstsq first. If too ill-conditioned, we can do a manual regularized solve.

            batch_size = A_block.shape[0]
            X = torch.empty((batch_size, n, 1), dtype=A_block.dtype, device=A_block.device)

            for b in range(batch_size):
                A_b = A_block[b]
                B_b = B_block[b]

                # Regularization step:
                # Compute AᵀA and AᵀB
                At = A_b.transpose(0,1) # shape [n,4]
                AtA = At.mm(A_b)        # shape [n,n]
                AtB = At.mm(B_b)        # shape [n,1]

                # Add small lambda * I to AtA
                AtA_reg = AtA + self.regularization * torch.eye(n, device=AtA.device)

                # Solve the regularized normal equations:
                # X_b = (AtA_reg)⁻¹ AtB
                # Using torch.linalg.solve:
                try:
                    X_b = torch.linalg.solve(AtA_reg, AtB)
                except RuntimeError:
                    # Fallback: use lstsq
                    X_b = torch.linalg.lstsq(AtA_reg, AtB).solution

                X[b] = X_b

            # Assign weights
            for i_node, w in enumerate(X):
                receiver = index_list[i_node]
                w = w.squeeze()
                weights[edges[1] == receiver] = w

        weights = weights.detach()
        return weights
