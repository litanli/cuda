# CUDA Playground
Work in progress collection of CUDA kernels, meant as a place to practice
CUDA programming on NVIDIA GPUs. 

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

*   Compile `nvcc -I./include -o vectoradd vectoradd.cu`
*   Run `./vectoradd`

Sometimes you can run into a mismatch between the Toolkit's PTX<sup>1</sup> 
version and your GPU's architecture generation. This causes kernel launches to 
fail (silently, unless we error check) with `cudaErrorUnsupportedPtxVersion`. 
If this is the case, compile and run `query.cpp`, which prints a number 
associated with your GPU's architecture:

*   `nvcc -o query query.cpp`
*   `./query` prints `Compute Capability: X.Y`
*   Add `-arch` flag to compile commmand: `nvcc -arch=sm_XY -o vectoradd vectoradd.cu`
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