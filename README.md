# CUDA Playground
Work in progress collection of CUDA kernels, meant as a place to practice
CUDA programming on NVIDIA GPUs. 

## Requirements
*   `nvcc` is the standard CUDA compiler provided by NVIDIA. It comes with
    the CUDA Toolkit. If you're using Linux or Windows WSL, follow the 
    installation guide here https://docs.nvidia.com/cuda/cuda-installation-guide-linux/ 
*   Once installed, add the following binaries directories to your
    `PATH` variable so your shell knows where to find `nvcc`.

    `export PATH=/usr/local/cuda/bin:$PATH` <br>
    `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`
    
    Add the above two lines to your `~/.bashrc` or `~/.zshrc` depending on
    which shell you're using, and then run `source ~/.bashrc` or 
    `source ~/.zshrc` to apply the changes.
*    Verify installation `nvcc --version`


## Build
The `include/helper_cuda.h` contains helper functions like status code
checking for CUDA runtime API function calls (`cudaMalloc`, `cudaMemcpy`, etc.).
In general, you should check the status codes and handle exceptions gracefully
(usually just `fprintf` to `stderr` and `exit(1)`). We tell `nvcc` to add
the `include` directory to places it can search for header files via `-I`.

`nvcc` conveniently follows `gcc` and `g++` convensions, so:

*   Compile `nvcc -I./include -o vectoradd vectoradd.c`
*   Run `./vectoradd`


## Contents at a Glance