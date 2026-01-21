using CUDA, BenchmarkTools

m = 512
k = 256
n = 128

A = CUDA.rand(Float32, m, k)
B = CUDA.rand(Float32, k, n)

C = CUDA.@sync begin
    @btime $A * $B
end
C_CPU = Array(C)

println("Result shape: ", size(C_CPU))
println("C[1,1] = ", C_CPU[1,1])
