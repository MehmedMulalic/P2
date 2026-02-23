import numpy as np
import time
import sys
from scipy import ndimage
from pathlib import Path

# Execution usage: ./file.py <nx> <ny>
SAVE_INTERVAL = 1667

A = 100.0              # Peak temperature (°C)
sigma_initial = 0.2    # Initial width (m)
x0, y0 = 0.5, 0.5      # Centre
alpha = 1.6563e-4      # Thermal diffusivity of pure silver (99.9%) (m^2/s)
Lx, Ly = 1.0, 1.0      # Domain limit (metres)

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

# 5-point stencil Laplacian kernel
kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]], dtype=np.float32) / dx**2

t0 = time.perf_counter()

for step in range(num_steps):
    laplacian = ndimage.convolve(u0, kernel, mode='reflect')
    u0 += alpha * dt * laplacian

    # if SAVE_INTERVAL > 0 and step % SAVE_INTERVAL == 0:
    #     u_current = u0.copy()
    #     current_time = step * dt

    #     snapshots.append( (current_time, u_current) )

t1 = time.perf_counter()

t = t1-t0
print(f"Time taken: {t}s")

u_final = u0.copy()
# snapshots.append( (t_final, u_final.copy()) )

filename_dir = Path(f"results")
filename_dir.mkdir(exist_ok=True)
filename = filename_dir / f"seq_stencil_explicit_{nx}_{ny}.csv"
np.savetxt(filename, u_final, delimiter=",")

# snapshot_dir = Path(f"snapshots")
# snapshot_dir.mkdir(exist_ok=True)
# for i, (t, u_snapshot) in enumerate(snapshots):
#     filename = snapshot_dir / f"seq_stencil_explicit_t{t:.3f}.csv"
#     np.savetxt(filename, u_snapshot, delimiter=",")
