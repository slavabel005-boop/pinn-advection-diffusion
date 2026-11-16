import deepxde as dde
import numpy as np 
import torch

dde.config.set_random_seed(0)

torch.set_default_device("cpu")

otrezok = dde.geometry.Interval(0, 1)

Pe = 5

def pde(x, c):

    dc_dx = dde.grad.jacobian(c, x, i = 0, j = 0)
    d2c_dx2 = dde.grad.hessian(c, x, i = 0, j = 0)

    r = d2c_dx2 - Pe * dc_dx

    return [r]

def c_exact(x):
    return (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1)

def boundary_inlet(x_0, on_boundary):
    return on_boundary and np.isclose(x_0[0], 0)

def boundary_outlet(x_1, on_boundary):
    return on_boundary and np.isclose(x_1[0], 1)


bc_inlet = dde.icbc.DirichletBC(otrezok, lambda x_0: 0.0, boundary_inlet)
bc_outlet = dde.icbc.DirichletBC(otrezok, lambda x_1: 1.0, boundary_outlet)

data = dde.data.PDE(
    otrezok,
    pde,
    [bc_inlet, bc_outlet],
    num_domain = 40,
    num_boundary = 2,
    num_test = 60,
    solution = c_exact,

)

net = dde.nn.FNN([1] + [32] * 3 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr = 0.001, metrics=["l2 relative error"])
model.train(iterations = 10000)

model.compile("L-BFGS")
model.train()

x_test = otrezok.uniform_points(1000, True)

c_pred = model.predict(x_test)

c_true = c_exact(x_test)

rel_l2 = np.linalg.norm(c_pred - c_true) / np.linalg.norm(c_true)

print(rel_l2)
