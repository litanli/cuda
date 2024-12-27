#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <functional>
#include <utility>

// Error checking macro
// See https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#error-handling
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

// Prints the execution time of a function that returns void, similar to a Python timing wrapper.
// See https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#using-cpu-timers
template <typename Func, typename... Args>
typename std::enable_if<std::is_void<decltype(std::declval<Func>()(std::declval<Args>()...))>::value, void>::type
time_exec(Func func, Args&&... args) {

    /*  The timed function may be asychronous, returning control immediately
        back to the calling CPU thread. To ensure proper timing:
        1) Call cudaDeviceSynchronize() to block CPU thread until GPU completes 
           any preceding tasks
        2) Record start time
        3) Call the timed function (which may be asynchronous)
        4) Synch again to block CPU thread until GPU completes
           the timed function
        5) Record end time
    */

    checkCudaErrors(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);  // place start event into default stream 0

    func(std::forward<Args>(args)...);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  // block host-side until stop event completes

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Execution took " << milliseconds << " ms" << std::endl;
}

// Measures the execution time of a function and returns the result of the
// function call
template <typename Func, typename... Args>
typename std::enable_if<!std::is_void<decltype(std::declval<Func>()(std::declval<Args>()...))>::value, decltype(std::declval<Func>()(std::declval<Args>()...))>::type
time_exec(Func func, Args&&... args) {
    auto tic = std::chrono::high_resolution_clock::now();
    auto result = func(std::forward<Args>(args)...);
    auto toc = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
    std::cout << "Execution took " << duration.count() << " ms" << std::endl;
    return result;
}