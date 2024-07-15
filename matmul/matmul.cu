#include <cmath>
#include <cassert>
// #include <cstdint>
#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std;

// Naive matmul kernel. Threads should cooperate to load row of A into shared
// memory.
__global__
void matmul_kernel(float* A, float* B, float* C, int N) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        
        float dot_prod = 0.0f;
        for (int i = 0; i < N; i++) {
            dot_prod += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = dot_prod;
    }
}

// Stub
void matmul(float* A_h, float* B_h, float* C_h, int N) {

    // Allocate device memory
    float *A_d, *B_d, *C_d;
    checkCudaErrors(cudaMalloc((void **) &A_d, N * N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &B_d, N * N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &C_d, N * N * sizeof(float)));

    // Copy input data to device global memory
    checkCudaErrors(cudaMemcpy(A_d, A_h, N * N * sizeof(float), cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(B_d, B_h, N * N * sizeof(float), cudaMemcpyDefault));

    dim3 grid_dim((N + 31) / 32, (N + 31) / 32);
    dim3 block_dim(32, 32);
 
    matmul_kernel<<<grid_dim, block_dim>>>(A_d, B_d, C_d, N);

    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());

    // Copy results back to host sychronously
    checkCudaErrors(cudaMemcpy(C_h, C_d, N * N * sizeof(float), cudaMemcpyDefault));

    // Free device memory
    checkCudaErrors(cudaFree(A_d));
    checkCudaErrors(cudaFree(B_d));
    checkCudaErrors(cudaFree(C_d));
}


int main(void) {
    int N = 3;
    float A_h[] = {1, 0, 0, 
                   0, 1, 0,
                   0, 0, 1};
    float B_h[] = {1, 1, 1,
                   1, 1, 1,
                   1, 1, 1};
    float C_h[N * N];

    matmul(A_h, B_h, C_h, N);
    // time_exec(matmul, A_h, B_h, C_h, N);

    // Verify
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert(C_h[i * N + j] == 1.0f);
        }
    }

    return 0;
}