#include <iostream>
#include <cublas_v2.h>

#define M 8
#define N 8
#define IDX2C(i, j, ld) (( (j)*(ld) ) + (i))

static __inline__ void modify(cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta) {
    cublasSscal(handle, n-q, &alpha, &m[IDX2C(p, q, ldm)], ldm);
    cublasSscal(handle, ldm-p, &beta, &m[IDX2C(p, q, ldm)], 1);
}

int main(void) {
    cublasHandle_t handle;
    cublasStatus_t status;
    cudaError_t cudaStatus;

    int i, j;
    float* devPtrA;
    float* a = 0;

    a = (float *)malloc(M * N * sizeof(*a));
    if (!a) {
        printf("Host memory allocation failed\n");
        free(a);
        return EXIT_FAILURE;
    }
    for (j = 0; j < N; ++j) {
        for (i = 0; i < M; ++i) {
            a[IDX2C(i, j, M)] = (float)(i*N + j+1);
        }
    }

    cudaStatus = cudaMalloc((void**)&devPtrA, M*N*sizeof(*a));
    if (cudaStatus != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS initialization failed\n");
        free(a);
        return EXIT_FAILURE;
    }

    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    status = cublasSetMatrix(M, N, sizeof(*a), a, M, devPtrA, M);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Data download failed\n");
        cudaFree(devPtrA);
        cublasDestroy(handle);
        free(a);
        return EXIT_FAILURE;
    }

    modify(handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);
    status = cublasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Data upload failed\n");
        cudaFree(devPtrA);
        cublasDestroy(handle);
        free(a);
        return EXIT_FAILURE;
    }

    cudaFree(devPtrA);
    cublasDestroy(handle);
    for (j = 0; j < N; ++j) {
        for (i = 0; i < M; ++i) {
            printf("%7.0f", a[IDX2C(i, j, M)]);
        }
        printf("\n");
    }
    free(a);
    return EXIT_SUCCESS;
}