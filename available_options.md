# Available options
* ~~sequential 5-point stencil (explicit Euler)~~
* sequential 5-point stencil with Laplace (explicit)
* sequential FFT (explicit)
* ~~Numba 5-point stencil (explicit)~~
* ~~Numba 5-point stencil with Laplace (explicit)~~
* Numba FFT (explicit)
* ~~CuPy 5-point stencil (explicit)~~
* ~~CuPy 5-point stencil with Laplace (explicit)~~
* ~~CuPy FFT (explicit)~~

## Notes
Numba with Laplace can never beat the cuSPARSE optimization which CuPy provides hence no implementation was made. However, this is a great opportunity for comparison to see how the regular stencil kernel performs compared to the sparse Laplace method using cuSPARSE. Both methods were tested for accuracy using the Neumann boundary condition and the results were accurate with an absolute tolerance of 1e-6.

# AI chats

## Laplace matrix
- https://chatgpt.com/c/694fcbca-9fe4-8325-afec-7bbde26d8d20
- https://chatgpt.com/c/694fd0d8-b8f0-832e-884d-3a3bd4f1f3b1

## Using CUDA in Numba
- https://claude.ai/chat/e3f64f0e-668c-44f6-b83f-02f7e0b82b82
- https://chatgpt.com/c/694fcde7-3aa8-8331-9219-1c534421af66

## FFT explanation
- https://chatgpt.com/c/695aa330-5a34-8328-a6e1-395b7ef29351
