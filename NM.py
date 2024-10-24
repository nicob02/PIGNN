from petsc4py import PETSc
from fenics import (NonlinearProblem, NewtonSolver, FiniteElement,
                    MixedElement, assemble, FunctionSpace, TestFunctions, Function,
                    interpolate, Expression, split, dot, inner, grad, dx, DirichletBC,
                    Constant, exp, ln, derivative, PETScKrylovSolver,
                    PETScFactory, near, PETScOptions, assign, File, plot, SpatialCoordinate)
from ufl import conditional
import numpy as np
import sys
import matplotlib.pyplot as plt
from core.geometry import ElectrodeMesh

class Problem(NonlinearProblem):
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)


class CustomSolver(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, Omega.mesh.mpi_comm(),
                              PETScKrylovSolver(), PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)
        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("pc_type", "ilu")
        self.linear_solver().set_from_options()

def outer_boundary(x):
    max_x = 1.0  
    max_y = 1.0  
    min_x = 0.0  
    min_y = 0.0  

    tol = 1E-16     # Mess around with this if needed

    right = near(x[0], max_x, tol)
    left = near(x[0], min_x, tol)
    up = near(x[1], max_y, tol)
    bottom = near(x[1], min_y, tol)

    #on_boundary = right or left or up or bottom
    on_boundary = right or up 
    return on_boundary

def electrode_surface(x):

    lb_electrode = [0.49, 0.45]
    ru_electrode = [0.51, 0.55]
    tol = 1E-5    # Mess around with this if needed

    right_electrode = (near(x[0], ru_electrode[0], tol) and (x[1] >= lb_electrode[1]) and (x[1] <= ru_electrode[1]))
    left_electrode  = (near(x[0], lb_electrode[0], tol) and (x[1] >= lb_electrode[1]) and (x[1] <= ru_electrode[1]))
    up_electrode    = (near(x[1], ru_electrode[1], tol) and (x[0] >= lb_electrode[0]) and (x[0] <= ru_electrode[0]))
    bottom_electrode= (near(x[1], lb_electrode[1], tol) and (x[0] >= lb_electrode[0]) and (x[0] <= ru_electrode[0]))

    on_electrode = right_electrode or left_electrode or up_electrode or bottom_electrode
    

    return on_electrode

Omega = ElectrodeMesh(ru=(1, 1), lb=(0, 0), density=100)

T_elem = FiniteElement("CG", Omega.mesh.ufl_cell(), 1)
V_elem = FiniteElement("CG", Omega.mesh.ufl_cell(), 2)       # Order 2 as voltage acts on a finer scale

ET_elem = MixedElement([V_elem, T_elem])
ET = FunctionSpace(Omega.mesh, ET_elem)

# Parameters 
rfa_params = [1060 , 3600 , 0.512 , 244000 , 310 , 0.33 , 0.02]  

Phi_test, Te_test = TestFunctions(ET)   # Test variables
u = Function(ET)                        # Solution variables
u0 = Function(ET)                       # Previous time step solutions


# Initial values
u  = interpolate(Constant((0, 310)), ET)
u0 = interpolate(Constant((0, 310)), ET)

Phi, Te = split(u)
Phi0, Te0 = split(u0)

# Starting and final time steps
t = 0.0
T = 80
dt = Expression('dtvalue', dtvalue = 0.1, degree=1)

x = SpatialCoordinate(Omega.mesh)

num_outer_boundary = 0
num_electrode_surface = 0

# Loop over all mesh nodes
for point in Omega.pos:
    
    if outer_boundary(point):
        num_outer_boundary += 1
    if electrode_surface(point):
        num_electrode_surface += 1

# Print the results
print(f"Number of nodes on the outer boundary: {num_outer_boundary}")
print(f"Number of nodes on the electrode surface: {num_electrode_surface}")


bc_bound_V = DirichletBC(ET.sub(0), Constant(0), outer_boundary)  # Volt = 0 at ground
bc_Temp = DirichletBC(ET.sub(1), Constant(310), outer_boundary)     # Temp = 310 at ground
bc_elec_V = DirichletBC(ET.sub(0), Constant(18), electrode_surface) # Volt = 18

sigma_electrode = Constant(1e8)  # Electrode conductivity (S/m)
sigma_liver = Constant(0.33)     # Liver conductivity (S/m)

sigma = 0.33*(1 + 0.02*(Te0-309))

F = ((-sigma)*(inner(grad(Phi),grad(Phi_test))))*dx             # Voltage residual

grad_phi = grad(Phi0)
squared_grad_v = dot(grad_phi, grad_phi)

Q = sigma*squared_grad_v
F += ((1060*3600*(Te-Te0)*Te_test)-(dt*Q*Te_test)+(dt*0.512*(inner(grad(Te),grad(Te_test)))))*dx    #Pennes model residual


J = derivative(F, u)
bcs = [bc_bound_V, bc_elec_V, bc_Temp]
problem = Problem(J, F, bcs)   
custom_solver = CustomSolver()

custom_solver.parameters["relative_tolerance"] = 1e-6  
custom_solver.parameters["maximum_iterations"] = 30

# Create VTK files for visualization output
vtkfile_Phi = File('solution_custom_solver/Phi.pvd')
vtkfile_Te = File('solution_custom_solver/Te.pvd')

while (t < T):
    print('Time:' + str(t))
    custom_solver.solve(problem, u.vector())
    u0.assign(u)
    _Phi, _Te = u.split()
    point = (0.3, 0.3)
    phi_value = _Phi(point)
    te_value = _Te(point)

    print('Volt at point {}: {}'.format(point, phi_value))
    print('Temp at point {}: {}'.format(point, te_value))
    vtkfile_Phi << (_Phi, t)
    vtkfile_Te << (_Te, t)   
    t += dt.dtvalue
    dt.dtvalue *= 1.05