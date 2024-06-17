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

@timing
def vec_add(a, b):
    return a + b

def clear_gpu_cache():
    to.cuda.empty_cache()
    assert to.cuda.memory_allocated() == 0
    assert to.cuda.memory_reserved() == 0  # cached memory
    
@timing
def main():

    # Test numpy and torch implementations of vector add,
    # clearning GPU cache in-between runs
    for n in range(200_000_000, 1_000_000_001, 200_000_000):

        print(f'--- n={n:,} ---')

        # numpy
        print("--- numpy cpu single-process")
        a, b = np.ones(n), np.ones(n)
        vec_add(a, b)
        del a, b
        gc.collect()

        # torch cpu
        print("--- torch cpu multiprocessing")
        a, b = to.ones(n), to.ones(n)
        print(a.device, b.device) 
        vec_add(a, b)  # to.Tensor CPU natively uses multiprocessing!
        del a, b
        gc.collect()
        clear_gpu_cache()

        # torch CUDA
        print("--- torch CUDA")
        a, b = to.ones(n).cuda(), to.ones(n).cuda()
        print(a.device, b.device)
        vec_add(a, b)
        del a, b
        gc.collect()
        clear_gpu_cache()


if __name__ == '__main__':
    main()