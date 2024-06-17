#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <functional>
#include <utility>

// Error checking macro
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

// Prints the execution time of a function, similar to a Python timing wrapper
template <typename Func, typename... Args>
typename std::enable_if<std::is_void<decltype(std::declval<Func>()(std::declval<Args>()...))>::value, void>::type
time_exec(Func func, Args&&... args) {
    auto tic = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto toc = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
    std::cout << "Execution took " << duration.count() << " ms" << std::endl;
}

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