from torch_geometric.data import Data
import torch
from collections import defaultdict
from tqdm import tqdm

class SolveWeightLST2d(object):
    '''
    Laplacian weights calculation via a local least-squares approach.

    Changes:
    - Attempt a simpler linear basis [x, y] before completely giving up.
    - If second-order basis [x, y, x², y²] fails, try the linear basis.
    - If linear basis also fails or insufficient neighbors, fallback to zero.
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
                before falling back to even simpler approximations.
        """
        self.regularization = regularization
        self.cond_warning = cond_warning
        self.max_reg_increase = max_reg_increase

        # Original higher-order basis and laplacian
        def func(pos):
            # Basis: [x, y, x², y²]
            x = pos[:, 0:1]
            y = pos[:, 1:2]
            v = torch.cat([x, y, x*x, y*y], dim=-1)
            return v

        def laplacian_func(pos):
            # Laplacian of [x, y, x², y²]:
            # x -> 0
            # y -> 0
            # x² -> 2
            # y² -> 2
            v = torch.zeros((pos.shape[0], 4), dtype=pos.dtype, device=pos.device)
            v[:, 2] = 2.0
            v[:, 3] = 2.0
            return v

        # Simpler linear basis and laplacian
        def func_linear(pos):
            # Basis: [x, y]
            x = pos[:, 0:1]
            y = pos[:, 1:2]
            v = torch.cat([x, y], dim=-1)
            return v

        def laplacian_func_linear(pos):
            # Laplacian of [x, y] is [0, 0]
            v = torch.zeros((pos.shape[0], 2), dtype=pos.dtype, device=pos.device)
            return v

        self.func = func
        self.laplacian_func = laplacian_func
        self.func_linear = func_linear
        self.laplacian_func_linear = laplacian_func_linear

    def __call__(self, data: Data):
        pos = data.pos
        edges = data.edge_index

        number_nodes = pos.shape[0]
        weights = torch.zeros_like(edges[1], dtype=torch.float)

        # Compute second-order basis and laplacian
        lap = self.laplacian_func(pos)
        diff_ = self.func(pos[edges[1]] - pos[edges[0]])

        # Attempt to solve for each node
        for i in tqdm(range(number_nodes)):
            diff = diff_[edges[1] == i]   # shape: [#neighbors,4]
            lap_val = lap[i:i+1]          # shape: [1,4]
            neibor = diff.shape[0]

            if neibor < 4:
                # Not enough neighbors for 4-term system, try linear basis
                # For linear basis, we need at least 2 neighbors
                if neibor < 2:
                    print("<4 Neighbours for node and also <2 for linear basis")
                    weights[edges[1] == i] = 0.0
                    continue
                # Attempt linear solve
                if not self.solve_linear_for_node(i, pos, edges, weights):
                    # If linear also fails, fallback to zero
                    weights[edges[1] == i] = 0.0
                continue

            A = diff.t()    # [4, #neighbors]
            B = lap_val.t() # [4,1]

            # Try solving second-order system
            X_b = self.attempt_solve(A, B)
            if X_b is None:
                # Second-order failed, try linear basis
                if not self.solve_linear_for_node(i, pos, edges, weights):
                    # If linear also fails, fallback to zero
                    weights[edges[1] == i] = 0.0
            else:
                # Assign weights
                # X_b: shape [#neighbors, 1]
                weights[edges[1] == i] = X_b.squeeze()

        return weights.detach()

    def attempt_solve(self, A, B):
        """
        Attempt to solve the given system A and B with increasing regularization.
        A shape: [basis_count, neighbors]
        B shape: [basis_count, 1]

        Returns:
            X_b: solution of shape [neighbors, 1] if successful
            None if failed
        """
        # Check condition number first
        try:
            U, S, Vt = torch.linalg.svd(A, full_matrices=False)
            cond_number = (S[0] / (S[-1] + 1e-14)).item()
            if cond_number > self.cond_warning:
                print(f"Warning: High condition number {cond_number}.")
        except RuntimeError as e:
            print(f"SVD failed: {e}")
            return None

        # Solve with regularization
        basis_count, neibor = A.shape
        return self.solve_with_regularization(A, B, neibor)

    def solve_with_regularization(self, A_b, B_b, n):
        reg = self.regularization
        max_reg = self.regularization * self.max_reg_increase

        At = A_b.transpose(0,1) # [n,basis_count]
        AtA = At.mm(A_b)        # [n,n]
        AtB = At.mm(B_b)        # [n,1]

        while reg <= max_reg:
            AtA_reg = AtA + reg * torch.eye(n, device=AtA.device)
            try:
                X_b = torch.linalg.solve(AtA_reg, AtB)
                # Check condition number again
                U, S, Vt = torch.linalg.svd(AtA_reg, full_matrices=False)
                cond_number = (S[0] / (S[-1] + 1e-14)).item()
                if cond_number > self.cond_warning:
                    reg *= 10
                    continue
                return X_b
            except RuntimeError:
                reg *= 10

        return None

    def solve_linear_for_node(self, i, pos, edges, weights):
        """
        Attempt a simpler linear basis [x, y] for node i before giving up.
        If linear fails, return False. If successful, assign weights and return True.

        For linear basis:
        - func_linear: [x,y]
        - laplacian_func_linear: [0,0]
        Therefore, Laplacian will be zero anyway, but we attempt a stable solve.

        We form a system:
            u(xj)-u(xi) ≈ [ (xj - xi), (yj - yi) ] · grad(u)
        The second derivatives vanish, so effectively this just stabilizes gradient calc.
        
        Since we're just approximating the Laplacian at this node and linear gives no second order term,
        we can directly set the laplacian weights to zero. However, let's follow the same solving logic.
        """

        node_mask = (edges[1] == i)
        node_neighbors = edges[0][node_mask]

        # At least 2 neighbors are needed for linear basis [x,y]
        if node_neighbors.shape[0] < 2:
            return False

        node_pos = pos[i:i+1]
        neigh_pos = pos[node_neighbors] - node_pos

        # Linear basis: [x, y]
        # Laplacian for linear basis is zero:
        # If you want to at least try to solve:
        A_lin = neigh_pos.t()  # [2, #neighbors]
        # laplacian_func_linear is [0,0], so B_lin is zero vector
        B_lin = torch.zeros((2, 1), device=pos.device)

        # Try to solve linear system:
        X_lin = self.attempt_solve(A_lin, B_lin)
        if X_lin is None:
            # Linear also failed
            return False

        # If linear succeeded, assign weights = X_lin and just ignore since laplacian is zero
        # Actually, since laplacian is zero for linear, we can just set weights to zero:
        weights[edges[1] == i] = 0.0
        return True
