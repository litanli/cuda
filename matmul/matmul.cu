#include <cassert>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <math.h>

using namespace std;

// Naive matmul kernel. 2 / 8 = 0.25 FLOPs/B.
__global__
void naive_matmul(float* A, float* B, float* C, int N) {
    
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

// Threads within same block cooperate to load tiles of A and tiles of B into 
// shared memory. 0.25 / (1/TILE_WIDTH) = 0.25 * TILE_WIDTH FLOPs/B. 
// A ∈ R^(I x J), B ∈ R^(J x K) and C ∈ R^(I x K).
#define TILE_WIDTH 32
__global__
void tiled_matmul(float* A, float* B, float* C, int I, int J, int K) {

    // Block scope, Grid lifetime
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];
    
    // Element of C that this thread is responsible for
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float val = 0.0f;
    for (int phase = 0; phase < J / TILE_WIDTH; phase++) {
        
        // Threads of block cooperatively load A and B tile into shared memory
        int A_col = phase * TILE_WIDTH + threadIdx.x;
        A_tile[threadIdx.y][threadIdx.x] = (row < I && A_col < J) ? A[row * J + A_col] : 0.0f;
        
        int B_row = phase * TILE_WIDTH + threadIdx.y;
        B_tile[threadIdx.y][threadIdx.x] = (B_row < J && col < K) ? B[B_row * K + col] : 0.0f;

        // Barrier to ensure all threads of block have loaded their portions 
        // of the tile. Read-after-write.
        __syncthreads();  

        // Compute dot product for current phase
        for (int i = 0; i < TILE_WIDTH; i++) {
            val += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }
        // Ensure all threads of block have finished using current A and B tile
        // before loading up the next two tiles. Write-after-read.
        __syncthreads();
    }
    
    // One element computed
    if (row < I && col < K) {
        C[row * K + col] = val;
    }
}

// Stub
void matmul(float* A_h, float* B_h, float* C_h, int I, int J, int K) {

    // Allocate device memory
    float *A_d, *B_d, *C_d;
    checkCudaErrors(cudaMalloc((void **) &A_d, I * J * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &B_d, J * K * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &C_d, I * K * sizeof(float)));

    // Copy input data to device global memory
    checkCudaErrors(cudaMemcpy(A_d, A_h, I * J * sizeof(float), cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(B_d, B_h, J * K * sizeof(float), cudaMemcpyDefault));


    dim3 grid_dim((std::max(J, K) + TILE_WIDTH - 1) / TILE_WIDTH, (std::max(I, J) + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    tiled_matmul<<<grid_dim, block_dim>>>(A_d, B_d, C_d, I, J, K);

    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());

    // Copy results back to host sychronously
    checkCudaErrors(cudaMemcpy(C_h, C_d, I * K * sizeof(float), cudaMemcpyDefault));

    // Free device memory
    checkCudaErrors(cudaFree(A_d));
    checkCudaErrors(cudaFree(B_d));
    checkCudaErrors(cudaFree(C_d));
}


int main(void) {
    int I = 64;
    int J = 64;
    int K = 64;

    float A_h[I * J];
    float B_h[J * K];
    float C_h[I * K];
    math::identity(A_h, I);    
    math::ones(B_h, J);
    matmul(A_h, B_h, C_h, I, J, K);
    // time_exec(matmul, A_h, B_h, C_h, I, J, K);

    // Verify
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < K; j++) {
            assert(C_h[i * K + j] == 1);
        }
    }
    return 0;
}