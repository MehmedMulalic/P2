import cupy as cp
import numpy as np
import time
import sys
from cupyx.scipy.fft import dctn, idctn
from pathlib import Path

NUM_EXECUTIONS = 1

def heat_equation(u0, alpha, dx, dt, num_steps, save_interval=10):
    N = u0.shape[0]
    
    # Frequency grid for DCT
    kx = cp.pi * cp.arange(N) / N
    ky = cp.pi * cp.arange(N) / N
    KX, KY = cp.meshgrid(kx, ky)
    
    # Eigenvalues for Neumann BC
    k_squared = (2 / dx**2) * (cp.cos(KX) + cp.cos(KY) - 2)
    
    # Transform to frequency space
    u_hat = dctn(u0, norm='ortho')
    
    # Evolution operator
    evolution_operator = cp.exp(alpha * k_squared * dt)

    # Animation snapshots as a tuple (t, u)
    snapshots = []
    snapshots.append( (0.0, cp.asnumpy(u0.copy())) )
    
    # Time stepping
    for step in range(num_steps):
        u_hat = u_hat * evolution_operator

        if save_interval > 0 and step % save_interval == 0:
            u_current = idctn(u_hat, norm='ortho').real
            current_time = step * dt

            snapshots.append( (current_time, cp.asnumpy(u_current.copy())) )
    
    # Transform back
    u_final = idctn(u_hat, norm='ortho').real
    final_time = num_steps * dt
    snapshots.append( (final_time, cp.asnumpy(u_final.copy())) )
    
    return u_final, snapshots

A = 100.0               # Peak temperature
sigma_initial = 0.05    # Initial width (0.05m)
x0, y0 = 0.5, 0.5       # Center
alpha = 1.172e-5        # Thermal diffusivity of steel (m^2/s)
Lx, Ly = 1.0, 1.0       # Domain limit (metres)

# Grid points
nx = int(sys.argv[1]) if len(sys.argv) > 1 else 256
ny = int(sys.argv[2]) if len(sys.argv) > 2 else 256
# nx, ny = 256, 256 # Grid points

dx = Lx / (nx - 1)      # x spacing
dy = Ly / (ny - 1)      # y spacing
dt = 0.1                # Time step
t_final = 60            # Final time step
num_steps = int(t_final / dt)

# NOT NEEDED FOR FFT
# r = alpha * dt / dx**2  # Stability
# if (r > 0.25):
#     raise Exception("Unstable simulation")

x = cp.linspace(0.0, Lx, nx)
y = cp.linspace(0.0, Ly, ny)
X, Y = cp.meshgrid(x, y)

# Initial condition (Gaussian bump)
u0 = A * cp.exp( -((X - x0)**2 + (Y - y0)**2) / (2 * sigma_initial**2) )

# Computation & D2H (including warm-up)
heat_equation(u0, alpha, dx, dt, num_steps)

t_avg: float = 0.0
for _ in range(NUM_EXECUTIONS):
    t0 = time.perf_counter()

    u_final, snapshots = heat_equation(u0, alpha, dx, dt, num_steps)
    u_cpu = cp.asnumpy(u_final)

    t1 = time.perf_counter()
    t_avg += t1-t0

t = t_avg / NUM_EXECUTIONS
print(f"Time taken: {t}s")

#TODO: define argument for saving results (and animation)
filename_dir = Path(f"results")
filename_dir.mkdir(exist_ok=True)
filename = filename_dir / "cupy_fft_{nx}_{ny}.csv"
# np.savetxt(filename, u_cpu, delimiter=",")

snapshot_dir = Path(f"snapshots")
snapshot_dir.mkdir(exist_ok=True)
for i, (t, u_snapshot) in enumerate(snapshots):
    filename = snapshot_dir / f"t{t:.3f}.csv"
    np.savetxt(filename, u_snapshot, delimiter=",")