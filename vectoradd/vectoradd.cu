#include <cmath>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <helper_cuda.h> 

using namespace std;

// __global__: declares a kernel: callable from host and device, executed on device
// __device__: declares a device function: callable from device, executed on device
// Vector add kernel running on GPU 
__global__
void vector_add_kernel(float *a, float *b, float *c, int n) {

    // blockIdx.x, blockDim.x, threadIdx.x are built-in variables, different 
    // threads see different values. Loop replaced by thread indexing.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // threads i >= n (if any) won't do any work
    if (i < n) {
        c[i] = a[i] + b[i];
        // printf("Thread %d: %f + %f = %f\n", i, a[i], b[i], c[i]);
    }
}

// Vector add stub (a stub is a function that calls a kernel)
void vector_add(float *a_h, float *b_h, float *c_h, int n) {
    
    size_t size = n * sizeof(float);

    // Allocate device memory, obtain pointers to them
    float *a_d, *b_d, *c_d;
    checkCudaErrors(cudaMalloc((void **) &a_d, size));
    checkCudaErrors(cudaMalloc((void **) &b_d, size));
    checkCudaErrors(cudaMalloc((void **) &c_d, size));
    
    // Copy data to device global memory
    checkCudaErrors(cudaMemcpy(a_d, a_h, size, cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(b_d, b_h, size, cudaMemcpyDefault));

    // Kernel launch (asynchronous) - config params <<<num blocks, num threads 
    // per block>>>. Max 1024 threads per block, threads per block should be 
    // multiple of 32 for efficiency reasons. 
    vector_add_kernel<<<(int)ceil(n / 1024), 1024>>>(a_d, b_d, c_d, n);

    // check for kernel launch errors
    checkCudaErrors(cudaGetLastError());  

    // Copy result in c_d back to host synchronously. 
    // See https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync
    checkCudaErrors(cudaMemcpy(c_h, c_d, size, cudaMemcpyDefault));

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);  
}


int main(void) {

    int n = 1 << 30;  // ~ 1 billion elements

    // Allocate host memory
    float *a_h = new float[n];
    float *b_h = new float[n];
    float *c_h = new float[n];

    for (int i=0; i < n; i++) {
        a_h[i] = 1.0f;
        b_h[i] = 2.0f;
    }

    // Whether vector_add computes on host or device is abstracted away
    // from the perspective of the host caller.
    // auto tic = chrono::high_resolution_clock::now();
    time_exec(vector_add, a_h, b_h, c_h, n);
    // auto toc = chrono::high_resolution_clock::now();
    // auto duration = chrono::duration_cast<chrono::milliseconds>(toc - tic);
    // cout << "vector_add took " << duration.count() << " ms" << endl;

    // Verify results
    for (int i=0; i < n; i++) {
        assert(c_h[i] == 3.0f);
    }
    
        
    delete[] a_h;
    delete[] b_h;
    delete[] c_h;
    return 0;
}