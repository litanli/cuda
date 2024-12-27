#include <cmath>
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


// Apply a Box filter to blur the RGB channels of an image. Each thread will 
// process one pixel across all 3 channels while ignoring the alpha channel.
// Note for FILTER_SIZE > 1, each thread should collaborate to load shared
// pixel values from shared memory. This kernel does not do so - making it a
// naive implementation that serves as a performance baseline.
__global__
void box_filter_kernel(uint8_t* in, uint8_t* out, int h, int w, int channels, 
                int filter_size) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < h && col < w) {
        // channels - 1 to skip the alpha channel
        for (int c=0; c<channels - 1; c++) {
            int pixel_val = 0;
            int pixel_count = 0;
            for (int blur_row=-filter_size/2; blur_row<=filter_size/2; blur_row++) {
                for (int blur_col=-filter_size/2; blur_col<=filter_size/2; blur_col++) {
                    int in_row = row + blur_row;
                    int in_col = col + blur_col;
                    if (in_row >= 0 && in_row < h && in_col >= 0 && in_col < w) {
                        int ofs_in = (in_row * w + in_col) * channels + c;
                        pixel_val += in[ofs_in];
                        pixel_count++;
                    }
                }
            }
            int ofs_out = (row * w + col) * channels + c;
            out[ofs_out] = (uint8_t)(pixel_val / pixel_count);
        }

        // Copy alpha channel
        int ofs_out = (row * w + col) * channels + 3;
        out[ofs_out] = in[ofs_out];
    }
}

// Stub
void box_filter(int filter_size, uint8_t* in_h, uint8_t* out_h, int h, int w, 
                int channels) {
    
    // Validate arguments
    if (filter_size % 2 == 0) {
        cerr << "Filter size must be odd." << endl;
        exit(1);
    }
    
    // Allocate device memory (channels - 1 because exclude alpha channel)
    uint8_t *in_d, *out_d;
    checkCudaErrors(cudaMalloc((void **) &in_d, h * w * channels));
    checkCudaErrors(cudaMalloc((void **) &out_d, h * w * channels));  

    // Copy input data to device global memory
    checkCudaErrors(cudaMemcpy(in_d, in_h, h * w * channels, cudaMemcpyDefault));

    dim3 grid_dim((w + 31)/32, (h + 31)/32);
    dim3 block_dim(32, 32);
    box_filter_kernel<<<grid_dim, block_dim>>>(in_d, out_d, h, w, channels, filter_size);

    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());

    // Copy results back to host sychronously
    checkCudaErrors(cudaMemcpy(out_h, out_d, h * w * channels, cudaMemcpyDefault));

    cudaFree(in_d);
    cudaFree(out_d);
}


int main(void) {

    // stbi_load returns image as 1D array in row major order, where each pixel 
    // is represented by 4 contiguous elements r, g, b, and alpha (channels=4).
    const char* filename = "../resources/tree.png";
    int h = 0, w = 0, channels = 0;
    uint8_t* image_h = stbi_load(filename, &w, &h, &channels, 0);
    if (image_h == NULL) {
        cerr << filename << " load failed." << endl;
        return 1;
    }

    cout << h << " " << w << " " << channels << endl;

    // Allocate host memory
    uint8_t* out_h = new uint8_t[h * w * channels];

    // Apply box filter, store results in OUT_H
    time_exec(box_filter, 7, image_h, out_h, h, w, channels);

    // Save results
    if (!stbi_write_png("tree_blur.png", w, h, channels, out_h, w*channels)) {
        cerr << "Save failed." << endl;
        stbi_image_free(image_h);
        delete[] out_h;
        return 1;
    }

    stbi_image_free(image_h);
    delete[] out_h;
}