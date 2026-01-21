import time
import sys
import numpy as np
from pathlib import Path

# Parameters
alpha = 0.01
Lx, Ly = 1.0, 1.0
nx = int(sys.argv[1]) if len(sys.argv[1]) > 0 else 256 # Grid points
ny = int(sys.argv[2]) if len(sys.argv[2]) > 0 else 256 # Grid points
# nx, ny = 256, 256
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

print(f"Executing kernel with grid points [{nx},{ny}]")

# Grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Time parameters
dt = 1e-6          # MUST be small for stability
t_final = 0.05
nt = int(t_final / dt)

# Initial condition (Gaussian pulse)
u = np.exp(-100 * ((X - 0.5)**2 + (Y - 0.5)**2))

# Storage (optional)
U = np.zeros((nt + 1, nx, ny))
U[0] = u.copy()

NUM_EXECUTION=1
t_avg: float = 0.0
for _ in range(NUM_EXECUTION):
    # Time stepping loop
    t0 = time.perf_counter()
    for n in range(nt):
        u_step = u.copy()

        # Laplacian (interior points only)
        laplacian = (
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        )

        # Euler update
        u_step[1:-1, 1:-1] += alpha * dt * laplacian

        # Neumann BC (zero gradient)
        u_step[0, 1:-1] = u_step[1, 1:-1]       # top
        u_step[-1, 1:-1] = u_step[-2, 1:-1]     # bottom
        u_step[1:-1, 0] = u_step[1:-1, 1]       # left
        u_step[1:-1, -1] = u_step[1:-1, -2]     # right

        # Corners (optional: average of neighbors)
        u_step[0, 0] = u_step[1, 0]
        u_step[0, -1] = u_step[1, -1]
        u_step[-1, 0] = u_step[-2, 0]
        u_step[-1, -1] = u_step[-2, -1]

        u = u_step
        U[n + 1] = u

    t1 = time.perf_counter()
    t_avg += t1-t0

t = t_avg / NUM_EXECUTION
print(f"Time taken: {t}s")

filename = Path(f"results/seq_stencil_explicit_{nx}_{ny}.csv")
if not filename.exists():
    np.savetxt(filename, U[-1], delimiter=",")