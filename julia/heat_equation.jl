using CUDA
using SparseArrays
using LinearAlgebra
using DelimitedFiles

# --- 1. Grid setup ---
nx, ny = 1000, 1000       # number of grid points
dx = 1.0 / nx
dy = 1.0 / ny
alpha = 1.0               # thermal diffusivity
dt = 0.1                   # timestep

N = nx * ny                # total number of unknowns

# --- 2. Initial condition: random 1000x1000 matrix ---
# u = 10 .* rand(nx, ny) .- 5
u = readdlm("../matrix.txt") |> x -> Float64.(x')
u_vec = Float64.(vec(u))   # flatten column-major for Kronecker product

# --- 3. Build 1D Laplacian ---
ex = ones(nx)
ey = ones(ny)
Lx = spdiagm(-1 => ex[2:end], 0 => -2*ex, 1 => ex[1:end-1]) / dx^2
Ly = spdiagm(-1 => ey[2:end], 0 => -2*ey, 1 => ey[1:end-1]) / dy^2

# Identity matrices
Ix = spdiagm(0 => ones(nx))
Iy = spdiagm(0 => ones(ny))

# --- 4. Build 2D Laplacian using Kronecker product ---
# L2D = kron(Iy, Lx) + kron(Ly, Ix)
L2D = kron(Iy, Lx) + kron(Ly, Ix)  # size N x N, sparse

# --- 5. Move to GPU ---
u_gpu = CuArray(u_vec)
L_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(L2D)

println("Laplacian size: ", size(L2D))
println("u vector length: ", length(u_vec))
println("L_gpu type: ", typeof(L_gpu))
println("u_gpu type: ", typeof(u_gpu))

# --- 6. Euler update on GPU ---
# u_next = u + dt * alpha * L * u
u_next_gpu = u_gpu + alpha * dt * (L_gpu * u_gpu)

# --- 7. Bring back to CPU and reshape ---
u_next = reshape(Array(u_next_gpu), nx, ny)

# --- 8. Display last 10 elements of flattened matrix ---
println("Next timestep temperature (last 10 elements):")
println(u_next[end-9:end])
