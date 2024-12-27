#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    return 0;
}