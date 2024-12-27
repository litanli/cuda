#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <helper_cuda.h>

// For png load and save
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;


// Convert image to grayscale using luminance = 0.21r + 0.72g + 0.07b.
// Both IN and OUT are 1D arrays in row-major order.
__global__
void grayscale_kernel(uint8_t* in, uint8_t* out, int h, int w, int channels) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < h && col < w) {
        
        // offset for luminance
        int ofs_out = row * w + col;

        // offset for image is larger by factor of CHANNELS
        int ofs_in = ofs_out * channels;  

        uint8_t r = in[ofs_in];
        uint8_t g = in[ofs_in + 1];
        uint8_t b = in[ofs_in + 2];
        // 4th channel image[ofs_image + 3] is alpha, not used in grayscale 
        // conversion

        out[ofs_out] = (uint8_t)(0.21f*r + 0.71f*g + 0.07f*b);
    }
}

// Stub
void grayscale(uint8_t* in_h, uint8_t* out_h, int h, int w, int channels) {

    // Allocate device memory (channels - 1 because exclude alpha channel)
    uint8_t *in_d, *out_d;
    checkCudaErrors(cudaMalloc((void **) &in_d, h * w * channels));
    checkCudaErrors(cudaMalloc((void **) &out_d, h * w));

    // Copy input data to device global memory
    checkCudaErrors(cudaMemcpy(in_d, in_h, h * w * channels, cudaMemcpyDefault));

    dim3 grid_dim((w + 31)/32, (h + 31)/32);
    dim3 block_dim(32, 32);
    grayscale_kernel<<<grid_dim, block_dim>>>(in_d, out_d, h, w, channels);

    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());

    // Copy results back to host sychronously
    checkCudaErrors(cudaMemcpy(out_h, out_d, h * w, cudaMemcpyDefault));

    cudaFree(in_d);
    cudaFree(out_d);
}


int main(void) {

    // stbi_load returns image as 1D array in row major order, where each pixel 
    // is represented by 4 contiguous elements r, g, b, and alpha (channels=4).
    const char* filename = "../resources/tree.png";
    int h = 0, w = 0, channels = 0;
    uint8_t* image_h = stbi_load(filename, &w, &h, &channels, 0);
    if (!image_h) {
        cerr << filename << " load failed." << endl;
        exit(1);
    }
    
    cout << h << " " << w << " " << channels << endl;

    // Allocate host memory
    uint8_t* lum_h = new uint8_t[h * w];

    // Calculate luminance, store results in LUM_H
    time_exec(grayscale, image_h, lum_h, h, w, channels);
    
    // Save results
    if (!stbi_write_png("tree_gray.png", w, h, 1, lum_h, w)) {
        cerr << "Save failed." << endl;
        stbi_image_free(image_h);
        delete[] lum_h;
        return 1;
    }

    stbi_image_free(image_h);
    delete[] lum_h;

    return 0;
}