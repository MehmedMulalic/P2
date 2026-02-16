import numpy as np
import cupy as cp
import sys
import time
from pathlib import Path
from cupyx.scipy import sparse

# Heat Equation -> du/dt = alpha * (d^2u/dx^2 + d^2u/dy^2)
# Heat Equation Laplace -> du/dt = alpha * delta^2U

alpha = 0.01 # Thermal diffusivity
Lx, Ly = 1.0, 1.0 # Domain limit [0, Lx], [0, Ly]
nx = int(sys.argv[1]) if len(sys.argv) > 1 else 256 # Grid points
ny = int(sys.argv[2]) if len(sys.argv) > 2 else 256 # Grid points
# nx, ny = 256, 256 # Grid points
dx = Lx / (nx - 1) # x spacing
dy = Ly / (ny - 1) # y spacing
dt = 1e-4 # Time step
t_final = 0.05 # Final simulation time step
num_steps = int(t_final / dt)

print(f"Executing kernel with grid points [{nx},{ny}]")

x = cp.linspace(0.0, Lx, nx)
y = cp.linspace(0.0, Ly, ny)
X, Y = cp.meshgrid(x, y)

# Initial condition (Gaussian pulse)
A = 1 # Base temperature 100C
u = A * cp.exp(-100 * ((X - 0.5)**2 + (Y - 0.5)**2))

# GPU data
Lx = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(nx, ny), format='csr')
Ly = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(nx, ny), format='csr')
Ix = sparse.eye(nx, format='csr')
Iy = sparse.eye(ny, format='csr')

# 2D Laplacian
L = (sparse.kron(Lx, Ix) + sparse.kron(Iy, Ly)) / dx**2
u_flat = u.flatten()

NUM_EXECUTIONS=20
t_avg: float = 0.0

for _ in range(NUM_EXECUTIONS):
    t0 = time.perf_counter()

    for step in range(num_steps):
        u_flat = u_flat + dt * alpha * (L @ u_flat)

    u_cpu = cp.asnumpy(u_flat).reshape((nx, ny))
    t1 = time.perf_counter()

    t_avg += (t1-t0)

t = t_avg / NUM_EXECUTIONS
print(f"Time taken: {t_avg}s")

filename = Path(f"results/cupy_laplace_{nx}_{ny}.csv")
# if not filename.exists():
np.savetxt(filename, u_cpu, delimiter=",")