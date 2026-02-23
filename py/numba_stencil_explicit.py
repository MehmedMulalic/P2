import numpy as np
import time
import sys
from numba import cuda
from pathlib import Path

SAVE_INTERVAL = 1667

@cuda.jit
def heat_equation(u, u_step, alpha, dx, dy, nx, ny, dt):
    i, j = cuda.grid(2)
    
    if 1 <= i < nx-1 and 1 <= j < ny-1:
        dudx2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2
        dudy2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2

        u_step[i, j] = u[i, j] + alpha * dt * (dudx2 + dudy2)

A = 100.0               # Peak temperature (°C)
sigma_initial = 0.2     # Initial width (m)
x0, y0 = 0.5, 0.5       # Centre
alpha = 1.6563e-4       # Thermal diffusivity of pure silver (99.9%) (m^2/s)
Lx, Ly = 1.0, 1.0       # Domain limit (metres)

# Grid points
nx = int(sys.argv[1]) if len(sys.argv) > 1 else 256
ny = int(sys.argv[2]) if len(sys.argv) > 2 else 256

dx = Lx / (nx - 1)      # x spacing
dy = Ly / (ny - 1)      # y spacing
dt = 0.0003             # Time step (s)
t_final = 60            # Final time step (s)
num_steps = int(t_final / dt)

# Stability
r = alpha * dt / dx**2
if (r > 0.25):
    raise Exception("Unstable simulation")

print(f"Executing kernel with grid points [{nx},{ny}]")

x = np.linspace(0.0, Lx, nx)
y = np.linspace(0.0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initial condition (Gaussian bump)
u0 = A * np.exp( -((X - x0)**2 + (Y - y0)**2) / (2 * sigma_initial**2) )

# Animation snapshots as a tuple (time, output)
# snapshots = []
# snapshots.append( (0.0, u0.copy()) )

threadsperblock = (16, 16)
blockspergrid_x = (nx + (threadsperblock[0] - 1)) // threadsperblock[0]
blockspergrid_y = (ny + (threadsperblock[1] - 1)) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

u_device = cuda.to_device(u0.copy())
u_step_device = cuda.to_device(u0.copy())

t0 = time.perf_counter()

for step in range(num_steps):
    heat_equation[blockspergrid, threadsperblock](u_device, u_step_device, alpha, dx, dy, nx, ny, dt)
    u_device, u_step_device = u_step_device, u_device

    # if SAVE_INTERVAL > 0 and step % SAVE_INTERVAL == 0:
    #     u_current = u_step_device.copy_to_host()
    #     current_time = step * dt

    #     snapshots.append( (current_time, u_current) )

u_cpu = u_device.copy_to_host()

t1 = time.perf_counter()

t = t1-t0
print(f"Time taken: {t}s")

# snapshots.append( (t_final, u_cpu) )

filename_dir = Path(f"results")
filename_dir.mkdir(exist_ok=True)
filename = filename_dir / f"numba_stencil_explicit_{nx}_{ny}.csv"
np.savetxt(filename, u_cpu, delimiter=",")

# snapshot_dir = Path(f"snapshots")
# snapshot_dir.mkdir(exist_ok=True)
# for i, (t, u_snapshot) in enumerate(snapshots):
#     filename = snapshot_dir / f"numba_stencil_explicit_t{t:.3f}.csv"
#     np.savetxt(filename, u_snapshot, delimiter=",")
