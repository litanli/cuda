from time import perf_counter
from functools import wraps
import numpy as np
import torch as to
import gc

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        tic = perf_counter()
        ret = f(*args, **kw)
        toc = perf_counter()
        print(f'{f.__name__} took {toc-tic:.3f} s')
        return ret
    return wrap

# @timing
def vec_add(a, b):
    c = [0] * len(a)
    for i in range(len(a)):
        c[i] = a[i] + b[i]
    return c

# @timing
def vec_add_np(a, b):
    return a + b

# @timing 
def vec_add_torch(a, b):
    return a + b

def memory_stats():
    print(to.cuda.memory_allocated()/1024**2)
    print(to.cuda.memory_reserved()/1024**2)  # cached memory

@timing
def main():

    runtimes_for = []
    runtimes_np = []
    runtimes_to_cpu = []
    runtimes_to_cuda = []

    for n in range(100_000_000, 1_000_000_001, 100_000_000):

        print(f'n={n}')
        
        # if n < 300_000_001:
        #     a = [1.0] * n
        #     b = [2.0] * n
        #     tic = perf_counter()
        #     vec_add(a, b)  # n=200e6 5.55s (single core)
        #     runtimes_for.append(perf_counter() - tic)

        #     del a, b
        #     gc.collect()

        a = np.ones(n)
        b = np.ones(n) * 2
        tic = perf_counter()
        vec_add_np(a, b)  # n=200e6 0.27s (single core)
        runtimes_np.append(perf_counter() - tic)
        
        del a, b
        gc.collect()

        a = to.ones(n)
        b = to.ones(n) * 2
        print(a.device, b.device) 
        tic = perf_counter()
        vec_add_torch(a, b)  # to.Tensor natively uses multiprocessing!
        runtimes_to_cpu.append(perf_counter() - tic)

        del a, b
        gc.collect()
        to.cuda.empty_cache()
        memory_stats()


        a = to.ones(n).cuda()
        b = (to.ones(n) * 2).cuda()
        print(a.device, b.device)
        tic = perf_counter()
        vec_add_torch(a, b)
        runtimes_to_cuda.append(perf_counter() - tic)

        del a, b
        gc.collect()
        to.cuda.empty_cache()
        memory_stats()

    print(runtimes_for)
    print(runtimes_np)
    print(runtimes_to_cpu)
    print(runtimes_to_cuda)

if __name__ == '__main__':
    main()