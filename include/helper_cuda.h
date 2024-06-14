#include <iostream>
#include <cuda_runtime.h>

// Helper functions for CUDA programming borrowed from 
// https://github.com/NVIDIA/cuda-samples

#define checkCudaErrors(val) check((val), __FILE__, __LINE__)

void check(cudaError_t err, const char *const file,
           int const line) {

    // cudaError_t are unsigned integers
    if (err != cudaSuccess) {
        fprintf(stderr, "%s at %s:%d\n", 
                cudaGetErrorName(err), file, line);
        exit(1);
    }
}