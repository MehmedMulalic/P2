import numpy as np
import cupy as cp
import time
import sys
from cupyx.scipy import ndimage
from pathlib import Path

alpha = 0.01
Lx, Ly = 1.0, 1.0
nx = int(sys.argv[1]) if len(sys.argv[1]) > 0 else 256 # Grid points
ny = int(sys.argv[2]) if len(sys.argv[2]) > 0 else 256 # Grid points
# nx, ny = 256, 256 # Grid points
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = 1e-4
t_final = 0.05
num_steps = int(t_final / dt)

print(f"Executing kernel with grid points [{nx},{ny}]")

x = cp.linspace(0.0, Lx, nx)
y = cp.linspace(0.0, Ly, ny)
X, Y = cp.meshgrid(x, y)

# Initial condition (Gaussian pulse)
u = cp.exp(-100 * ((X - 0.5)**2 + (Y - 0.5)**2))

# 5-point stencil Laplacian kernel
kernel = cp.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]], dtype=cp.float32) / dx**2

NUM_EXECUTIONS = 20
t_avg: float = 0.0
for _ in range(NUM_EXECUTIONS):
    t0 = time.perf_counter()
    for n in range(num_steps):
        laplacian = ndimage.convolve(u, kernel)
        u = u + alpha * dt * laplacian

    u_final = cp.asnumpy(u)
    t1 = time.perf_counter()

    t_avg += t1-t0

t = t_avg / NUM_EXECUTIONS
print(f"Time taken: {t_avg}s")

# Save to CSV
filename = Path(f"results/cupy_stencil_explicit_{nx}_{ny}.csv")
if not filename.exists():
    np.savetxt(filename, u_final, delimiter=",")
