import numpy as np
import time
import sys
from scipy.fft import dctn, idctn
from pathlib import Path

# Execution usage: ./file.py <nx> <ny>
NUM_EXECUTIONS = 1

def heat_equation(u0, alpha, dx, dt, num_steps, save_interval=100):
    N = u0.shape[0]
    
    # Frequency grid for DCT
    kx = np.pi * np.arange(N) / N
    ky = np.pi * np.arange(N) / N
    KX, KY = np.meshgrid(kx, ky)
    
    # Eigenvalues for Neumann BC
    k_squared = (2/dx**2) * (np.cos(KX) + np.cos(KY) - 2)
    
    u_hat = dctn(u0, norm='ortho')
    evolution_operator = np.exp(alpha * k_squared * dt)
    
    # Animation snapshots as a tuple (time, output)
    snapshots = []
    snapshots.append( (0.0, u0.copy()) )

    for step in range(num_steps):
        u_hat = u_hat * evolution_operator

        if save_interval > 0 and step % save_interval == 0:
            u_current = idctn(u_hat, norm='ortho').real
            current_time = step * dt

            snapshots.append( (current_time, u_current.copy()) )
    
    u_final = idctn(u_hat, norm='ortho')
    final_time = num_steps * dt
    snapshots.append( (final_time, u_final.copy()) )

    return u_final, snapshots

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
dt = 0.005               # Time step (s)
t_final = 60            # Final time step (s)
num_steps = int(t_final / dt)

# Stability not needed for FFT since it is unconditionally stable
print(f"Executing kernel with grid points [{nx},{ny}]")

x = np.linspace(0.0, Lx, nx)
y = np.linspace(0.0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initial condition (Gaussian bump)
u0 = A * np.exp( -((X - x0)**2 + (Y - y0)**2) / (2 * sigma_initial**2) )

t_avg: float = 0.0
for _ in range(NUM_EXECUTIONS):
    t0 = time.perf_counter()

    u_final, snapshots = heat_equation(u0, alpha, dx, dt, num_steps)

    t1 = time.perf_counter()
    t_avg += t1-t0

t = t_avg / NUM_EXECUTIONS
print(f"Time taken: {t}s")

filename_dir = Path(f"results")
filename_dir.mkdir(exist_ok=True)
filename = filename_dir / f"seq_fft_{nx}_{ny}.csv"
np.savetxt(filename, u_final, delimiter=",")

snapshot_dir = Path(f"snapshots")
snapshot_dir.mkdir(exist_ok=True)
for i, (t, u_snapshot) in enumerate(snapshots):
    filename = snapshot_dir / f"seq_fft_t{t:.3f}.csv"
    np.savetxt(filename, u_snapshot, delimiter=",")
