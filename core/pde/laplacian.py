from torch_geometric.data import Data
import torch
from collections import defaultdict
from tqdm import tqdm

class SolveWeightLST2d(object):
    '''
    Laplacian weights calculation via a local least-squares approach.

    Changes:
    - Removed the cross term x*y from the basis to reduce potential ill-conditioning.
    - Added condition number checks.
    - Implemented dynamic Tikhonov regularization increase if the system is ill-conditioned.
    - If still ill-conditioned, fallback to a simpler linear approximation (no second order terms),
      thus setting Laplacian approximation to zero at that node.
    '''

    def __init__(self, regularization=1e-10, cond_warning=1e7, max_reg_increase=1e5):
        """
        Parameters:
            regularization: float
                Initial small Tikhonov regularization parameter.
            cond_warning: float
                Condition number threshold above which a warning is printed.
            max_reg_increase: float
                Maximum factor by which we can increase the regularization
                before falling back to a simpler approximation.
        """
        self.regularization = regularization
        self.cond_warning = cond_warning
        self.max_reg_increase = max_reg_increase

        def func(pos):
            # Basis: [x, y, x², y²]
            x = pos[:, 0:1]
            y = pos[:, 1:2]
            v = torch.cat([x, y, x*x, y*y], dim=-1)
            return v

        def laplacian_func(pos):
            # Laplacian of [x, y, x², y²]:
            # x -> ∆x = 0
            # y -> ∆y = 0
            # x² -> ∆x² = 2
            # y² -> ∆y² = 2
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
        diff_ = self.func(pos[edges[1]] - pos[edges[0]])

        all_A_dict = defaultdict(list)
        all_B_dict = defaultdict(list)
        index_dict = defaultdict(list)

        # Build local systems
        for i in tqdm(range(number_nodes)):
            diff = diff_[edges[1] == i]       # shape: [#neighbors, 4]
            laplacian_value = lap[i:i+1]      # shape: [1,4]

            A = diff.t()  # A: [4, #neighbors]
            B = laplacian_value.t()  # B: [4,1]
            neibor = A.shape[1]

            if neibor < 4:
                # Not enough neighbors to solve for a 4-term system
                # We'll skip or fallback to zero (no meaningful Laplacian)
                print("<4 Neighbours for node")
                # Set all weights for this node to zero (fallback)
                # edges[1] == i gives the edges connected to this node
                weights[edges[1] == i] = 0.0
                continue

            # Check condition number
            try:
                U, S, Vt = torch.linalg.svd(A, full_matrices=False)
                cond_number = (S[0] / (S[-1] + 1e-14)).item()
                if cond_number > self.cond_warning:
                    print(f"Warning: High condition number {cond_number} at node {i} with {neibor} neighbors.")
            except RuntimeError as e:
                print(f"SVD failed for node {i}: {e}")
                # Fallback to zero Laplacian
                weights[edges[1] == i] = 0.0
                continue

            all_A_dict[neibor].append((A, B, i))

        # Solve local systems by group of same neighbor count
        for n in all_A_dict.keys():
            A_list = [item[0] for item in all_A_dict[n]]
            B_list = [item[1] for item in all_A_dict[n]]
            index_list = [item[2] for item in all_A_dict[n]]

            A_block = torch.stack(A_list, dim=0) # [batch, 4, n]
            B_block = torch.stack(B_list, dim=0) # [batch, 4, 1]
            batch_size = A_block.shape[0]

            X = torch.empty((batch_size, n, 1), dtype=A_block.dtype, device=A_block.device)
            for b in range(batch_size):
                A_b = A_block[b]
                B_b = B_block[b]

                # Solve the system with the current regularization
                X_b = self.solve_with_regularization(A_b, B_b, n)
                if X_b is None:
                    # If we still can't solve it properly, fallback:
                    # Set Laplacian to zero
                    receiver = index_list[b]
                    weights[edges[1] == receiver] = 0.0
                else:
                    X[b] = X_b

            # Assign weights if we got a solution
            for i_node, receiver in enumerate(index_list):
                # If we fell back to zero, we already assigned weights
                # Otherwise, assign computed weights
                if torch.all(weights[edges[1] == receiver] == 0.0):
                    # Already set to zero (fallback)
                    continue
                w = X[i_node].squeeze()
                weights[edges[1] == receiver] = w

        weights = weights.detach()
        return weights

    def solve_with_regularization(self, A_b, B_b, n):
        """
        Attempt to solve the normal equations with increasing regularization if necessary.
        If after several attempts condition number remains too high or solution fails,
        return None to indicate fallback should occur.
        """
        reg = self.regularization
        max_reg = self.regularization * self.max_reg_increase

        At = A_b.transpose(0,1) # [n,4]
        AtA = At.mm(A_b)        # [n,n]
        AtB = At.mm(B_b)        # [n,1]

        while reg <= max_reg:
            AtA_reg = AtA + reg * torch.eye(n, device=AtA.device)
            # Attempt to solve
            try:
                X_b = torch.linalg.solve(AtA_reg, AtB)
                # Check condition number again after adding reg
                # Use SVD on AtA_reg:
                U, S, Vt = torch.linalg.svd(AtA_reg, full_matrices=False)
                cond_number = (S[0] / (S[-1] + 1e-14)).item()
                if cond_number > self.cond_warning:
                    # Increase regularization and try again
                    reg *= 10
                    continue
                return X_b
            except RuntimeError:
                # Increase reg and try again
                reg *= 10

        # If we reached here, even with increased regularization we failed
        return None
