import numpy as np
import time
import sys
from scipy.fft import dctn, idctn
from pathlib import Path

def heat_equation_dct_cupy(u0, alpha, dx, dt, num_steps):
    N = u0.shape[0]
    
    # Frequency grid for DCT
    kx = np.pi * np.arange(N) / N
    ky = np.pi * np.arange(N) / N
    KX, KY = np.meshgrid(kx, ky)
    
    # k^2 for Neumann BC
    k_squared = (2/dx**2) * (np.cos(KX) + np.cos(KY) - 2)
    
    # Transform to frequency space
    u_hat = dctn(u0, norm='ortho')
    
    # Evolution operator
    evolution_operator = np.exp(alpha * k_squared * dt)
    
    # Time stepping
    for _ in range(num_steps):
        u_hat = u_hat * evolution_operator
    
    # Transform back
    u_final = idctn(u_hat, norm='ortho')
    
    return u_final

alpha = 0.01 # Thermal diffusivity
Lx, Ly = 1.0, 1.0 # Domain limit [0, Lx], [0, Ly]
nx = int(sys.argv[1]) if len(sys.argv[1]) > 0 else 256 # Grid points
ny = int(sys.argv[2]) if len(sys.argv[2]) > 0 else 256 # Grid points
# nx, ny = 256, 256 # Grid points
dx = Lx / (nx - 1) # x spacing
dy = Ly / (ny - 1) # y spacing
dt = 1e-4 # Time step
t_final = 0.05 # Final simulation time step
num_steps = int(t_final / dt)

x = np.linspace(0.0, Lx, nx)
y = np.linspace(0.0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initial condition (Gaussian pulse)
u0 = np.exp(-100 * ((X - 0.5)**2 + (Y - 0.5)**2))

# Computation
NUM_EXECUTIONS = 20
t_avg: float = 0.0
for _ in range(NUM_EXECUTIONS):
    t0 = time.perf_counter()

    u_final = heat_equation_dct_cupy(u0, alpha, dx, dt, num_steps)

    t1 = time.perf_counter()
    t_avg += t1-t0

t = t_avg / NUM_EXECUTIONS
print(f"Time taken: {t}s")

filename = Path(f"results/seq_fft_{nx}_{ny}.csv")
if not filename.exists():
    np.savetxt(filename, u_final, delimiter=",")