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


// Apply a Box filter to blur an image.
// Note for FILTER_SIZE > 1, each thread should collaborate to load shared
// pixel values from shared memory. This kernel does not do so - making it a
// naive implementation that serves as a performance baseline.
__global__
void box_filter(uint8_t* in, uint8_t* out, int h, int w, int channels, 
                int filter_size) {
    
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_row < h && out_col < w) {
        int pixel_val = 0;
        int pixel_count = 0;

        for (int blur_row=-filter_size/2; blur_row<=filter_size/2; blur_row++) {
            for (int blur_col=-filter_size/2; blur_col<=filter_size/2; blur_col++) {
                int in_row = out_row + blur_row;
                int in_col = out_col + blur_col;
                if (in_row >= 0 && in_row < h && in_col >= 0 && in_col < w) {
                    pixel_val += in[in_row*w + in_col];
                    pixel_count++;
                }
            }
        }
        out[out_row*w + out_col] = (uint8_t)(pixel_val / pixel_count);
    }
}


// Stub
void box_filter(int filter_size) {
    if (filter_size % 2 == 0) {
        cerr << "Filter size must be odd." << endl;
        exit(1);
    }
}