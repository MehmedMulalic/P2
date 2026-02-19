import numpy as np
import cupy as cp
import time
import sys
from cupyx.scipy import sparse
from pathlib import Path

# Execution usage: ./file.py <nx> <ny>
NUM_EXECUTIONS = 1
SAVE_INTERVAL = 100

A = 100.0               # Peak temperature (°C)
sigma_initial = 0.2    # Initial width (m)
x0, y0 = 0.5, 0.5       # Centre
alpha = 1.6563e-4        # Thermal diffusivity of pure silver (99.9%) (m^2/s)
Lx, Ly = 1.0, 1.0       # Domain limit (metres)

# Grid points
nx = int(sys.argv[1]) if len(sys.argv) > 1 else 256
ny = int(sys.argv[2]) if len(sys.argv) > 2 else 256

dx = Lx / (nx - 1)      # x spacing
dy = Ly / (ny - 1)      # y spacing
dt = 0.005              # Time step (s)
t_final = 60          # Final time step (s)
num_steps = int(t_final / dt)

# Stability
r = alpha * dt / dx**2
if (r > 0.25):
    raise Exception("Unstable simulation")

print(f"Executing kernel with grid points [{nx},{ny}]")

x = cp.linspace(0.0, Lx, nx)
y = cp.linspace(0.0, Ly, ny)
X, Y = cp.meshgrid(x, y)

# Initial condition (Gaussian bump)
u0 = A * cp.exp( -((X - x0)**2 + (Y - y0)**2) / (2 * sigma_initial**2) )

# GPU data
Lx = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(nx, ny), format='csr')
Ly = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(nx, ny), format='csr')
Ix = sparse.eye(nx, format='csr')
Iy = sparse.eye(ny, format='csr')

# 2D Laplacian
L = (sparse.kron(Lx, Ix) + sparse.kron(Iy, Ly)) / dx**2
u_flat = u0.flatten()

# Animation snapshots as a tuple (time, output)
snapshots = []
snapshots.append( (0.0, cp.asnumpy(u0.copy())) )

t_avg: float = 0.0
for _ in range(NUM_EXECUTIONS):
    t0 = time.perf_counter()

    for step in range(num_steps):
        u_flat = u_flat + dt * alpha * (L @ u_flat)

        if SAVE_INTERVAL > 0 and step % SAVE_INTERVAL == 0:
            u_current = cp.asnumpy(u_flat.copy()).reshape((nx, ny))
            current_time = step * dt

            snapshots.append( (current_time, u_current) )

    u_cpu = cp.asnumpy(u_flat).reshape((nx, ny))

    t1 = time.perf_counter()
    t_avg += t1-t0

t = t_avg / NUM_EXECUTIONS
print(f"Time taken: {t}s")

snapshots.append( (t_final, u_cpu.copy()) )

filename_dir = Path(f"results")
filename_dir.mkdir(exist_ok=True)
filename = filename_dir / f"cupy_laplace_{nx}_{ny}.csv"
np.savetxt(filename, u_cpu, delimiter=",")

snapshot_dir = Path(f"snapshots")
snapshot_dir.mkdir(exist_ok=True)
for i, (t, u_snapshot) in enumerate(snapshots):
    filename = snapshot_dir / f"cupy_laplace_t{t:.3f}.csv"
    np.savetxt(filename, u_snapshot, delimiter=",")
