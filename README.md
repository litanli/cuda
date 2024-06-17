# CUDA Playground
Work in progress collection of CUDA kernels, meant as a place to practice
CUDA programming on NVIDIA GPUs. A lot of these kernels will follow those found 
in [Programming Massively Parallel Processors](https://lnkd.in/gYgRFdGW) except
that I've added error checking and tried to write cleaner implementations where
possible. For more kernel examples, NVIDIA's [cuda-samples](https://github.com/NVIDIA/cuda-samples)
repo contains a large collection of code.

CUDA kernel implementations are low level so they require some
basic knowledge of GPU architecture for implementation and performance 
profiling, especially of the GPU's memory hierarchy. GPU hardware architecture 
and the CUDA API evolve over time, however fundamental concepts should remain 
fairly static and are worth learning. Some of these include memory hierarchy, 
level-caching, logical organization (grids, blocks, warps, threads) and 
physical organization (CUDA cores, Tensor cores, SMs). All of this can be found 
in the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).

## Requirements
*   NVDIA GPU, see https://developer.nvidia.com/cuda-gpus for supported devices.
*   `nvcc` is the standard CUDA compiler provided by NVIDIA. It comes with
    the CUDA Toolkit, and compiles both host C++ and device CUDA code. If 
    you're using Linux or Windows WSL, follow the installation guide here 
    https://docs.nvidia.com/cuda/cuda-installation-guide-linux/ 
*   Once installed, add the following binary directories to your
    `PATH` variable so your shell knows where to find `nvcc`.

    `export PATH=/usr/local/cuda/bin:$PATH` <br>
    `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`
    
    Add the above two lines to your `~/.bashrc` or `~/.zshrc` depending on
    which shell you're using, and then run `source ~/.bashrc` or 
    `source ~/.zshrc` to apply the changes.
*    Verify installation `nvcc --version`


## Compile


The `include/helper_cuda.h` header file contains the `check` helper function 
for checking status codes returned by CUDA API function calls (`cudaMalloc`, 
`cudaMemcpy`, etc.) and handling exceptions gracefully (just `fprintf` to 
`stderr` and `exit(1)`). Tell `nvcc` to add the `include` directory to places 
it can search for header files via `-I`.

`nvcc` conveniently follows `gcc` and `g++` command line syntax, so:

*   Compile `nvcc -I ../include -o vectoradd vectoradd.cu`
*   Run `./vectoradd`

Sometimes you can run into a mismatch between the Toolkit's PTX<sup>1</sup> 
version and your GPU's architecture generation. This causes kernel launches to 
fail (silently, unless we error check) with `cudaErrorUnsupportedPtxVersion`. 
If this is the case, compile and run `query.cpp`, which prints a number 
associated with your GPU's architecture:

*   `nvcc -o query query.cpp`
*   `./query` prints `Compute Capability: X.Y`
*   Add `-arch` flag to compile commmand: `nvcc -arch=sm_XY -I ../include -o vectoradd vectoradd.cu`
    and you should be good to go.

<sup>1</sup>PTX stands for Parallel Thread Execution. Whereas CPUs follow the 
x86 or ARM instruction set, NVIDIA GPUs follow the PTX ISA. An instruction set
architecture (ISA) is neither software nor hardware - it's a specification
of low-level instructions that a device should support, which is implemented in 
machine code (1's and 0's). For host/CPU C++ code, nvcc generates Assembly code
just like g++ and clang++. For device/GPU CUDA code, it generates PTX code 
which invokes machine code implementing the PTX ISA. Thus PTX refers to both 
the ISA that NIVIDA GPUs follow and the "Assembly code" for it.

## Contents at a Glance
*   vectoradd - sum two vectors 
*   grayscale - converts rgba png into grayscale/luminance values



