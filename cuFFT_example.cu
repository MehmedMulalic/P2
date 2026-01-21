#include <stdio.h>
#include <cufft.h>

int main() {
    const int N = 8;

    // Allocate host memory
    cufftComplex h_data[N];
    for (int i = 0; i < N; i++) {
        h_data[i].x = i; // real part
        h_data[i].y = 0; // imaginary part
    }

    // Allocate device memory
    cufftComplex *d_data;
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * N);
    cudaMemcpy(d_data, h_data, sizeof(cufftComplex) * N, cudaMemcpyHostToDevice);

    // Create cuFFT plan
    cufftHandle plan;
    if (cufftPlan1d(&plan, N, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed\n");
        return -1;
    }

    // Execute FFT (forward)
    if (cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecC2C failed\n");
        return -1;
    }

    // Copy results back to host
    cudaMemcpy(h_data, d_data, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost);

    // Print results
    printf("FFT result:\n");
    for (int i = 0; i < N; i++) {
        printf("%d: (%f, %f)\n", i, h_data[i].x, h_data[i].y);
    }

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_data);

    return 0;
}
