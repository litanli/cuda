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
#define TILE_WIDTH 32
__global__
void tiled_matmul(float* A, float* B, float* C, int N) {

    // Block scope, Grid lifetime
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];
    
    // Element of C that this thread is responsible for
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float val = 0.0f;
    for (int phase = 0; phase < N / TILE_WIDTH; phase++) {
        
        // Threads of block cooperatively load A and B tile into shared memory
        if (row < N && phase * TILE_WIDTH + threadIdx.x < N) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * N + phase * TILE_WIDTH + threadIdx.x];
        } else {
            // 0.0f won't affect dot product
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;  
        }
        if ((phase * TILE_WIDTH + threadIdx.y) < N && col < N) {
            B_tile[threadIdx.y][threadIdx.x] = B[(phase * TILE_WIDTH + threadIdx.y) * N + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Barrier to ensure all threads of block have loaded their portions 
        // of the tile. Read-after-write.
        __syncthreads();  

        // Compute dot product for current phase using values cooperatively 
        // loaded into shared memory
        for (int i = 0; i < TILE_WIDTH; i++) {
            val += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }
        // Ensure all threads of block have finished using current A and B tile
        // before loading up the next two tiles. Write-after-read.
        __syncthreads();
    }
    
    // One element computed
    if (row < N && col < N) {
        C[row * N + col] = val;
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

    // dim3 grid_dim((N + 31) / 32, (N + 31) / 32);
    // dim3 block_dim(32, 32);
    // matmul_kernel<<<grid_dim, block_dim>>>(A_d, B_d, C_d, N);

    dim3 grid_dim((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    tiled_matmul<<<grid_dim, block_dim>>>(A_d, B_d, C_d, N);

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
    int N = 64;

    float A_h[N * N];
    float B_h[N * N];
    float C_h[N * N];
    math::identity(A_h, N);    
    math::ones(B_h, N);
    matmul(A_h, B_h, C_h, N);
    // time_exec(matmul, A_h, B_h, C_h, N);
    // Verify
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert(C_h[i * N + j] == 1);
        }
    }


    

    

    return 0;
}