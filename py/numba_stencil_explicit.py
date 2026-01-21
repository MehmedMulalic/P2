import numpy as np
import time
import sys
from pathlib import Path
from numba import cuda

alpha = 0.01
Lx, Ly = 1.0, 1.0
nx = int(sys.argv[1]) if len(sys.argv[1]) > 0 else 256 # Grid points
ny = int(sys.argv[2]) if len(sys.argv[2]) > 0 else 256 # Grid points
# nx, ny = 256, 256
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = 1e-4
t_final = 0.05
num_steps = int(t_final / dt)

print(f"Executing kernel with grid points [{nx},{ny}]")

x = np.linspace(0.0, Lx, nx)
y = np.linspace(0.0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initial condition (Gaussian pulse)
u = np.exp(-100 * ((X - 0.5)**2 + (Y - 0.5)**2)) # U_n
u_step = u.copy() # U_n+1

@cuda.jit
def heat_step(u, u_step, alpha, dx, dy, nx, ny, dt):
    i, j = cuda.grid(2)
    if 1 <= i < nx-1 and 1 <= j < ny-1:
        dudx2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2
        dudy2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
        u_step[i, j] = u[i, j] + alpha * dt * (dudx2 + dudy2)

threadsperblock = (16, 16)
blockspergrid_x = (nx + (threadsperblock[0] - 1)) // threadsperblock[0]
blockspergrid_y = (ny + (threadsperblock[1] - 1)) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

u_device = cuda.to_device(u)
u_step_device = cuda.to_device(u_step)

NUM_EXECUTIONS=20
t_avg: float = 0.0
for _ in range(NUM_EXECUTIONS):
    t0 = time.perf_counter()
    for n in range(num_steps):
        heat_step[blockspergrid, threadsperblock](u_device, u_step_device, alpha, dx, dy, nx, ny, dt)
        u_device, u_step_device = u_step_device, u_device

    u_final = u_device.copy_to_host()
    t1 = time.perf_counter()
    t_avg += t1-t0

t = t_avg / NUM_EXECUTIONS
print(f"Time taken: {t}s")

filename = Path(f"results/numba_stencil_{nx}_{ny}.csv")
if not filename.exists():
    np.savetxt(filename, u_final, delimiter=",")