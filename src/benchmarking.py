# In src/benchmarking.py
import time
import numpy as np
import pandas as pd
from .algorithms import (
    naive_algorithm,
    optimized_standard_algorithm,
    strassens_algorithm,
    strassen_padding,
    block_multiply_algorithm,
    parallel_naive_algorithm,
    hybrid_strassen_padding
)

def time_algorithm(func, A, B, **kwargs):
    """Times a single execution of a matrix multiplication function."""
    start_time = time.perf_counter()
    func(A, B, **kwargs)
    end_time = time.perf_counter()
    return end_time - start_time

def run_benchmark_suite(matrix_sizes, block_size, num_threads, hybrid_threshold):
    """Runs all algorithms over a range of matrix sizes and collects timing data."""
    results = []
    
    # Using the function names from your src/algorithms.py file
    algorithms_to_benchmark = {
        "Naive Standard": naive_algorithm,
        "NumPy Optimized": optimized_standard_algorithm,
        "Strassen (Padded)": strassen_padding,
        "Block Multiply": block_multiply_algorithm,
        "Parallel Naive": parallel_naive_algorithm,
        "Hybrid Strassen": hybrid_strassen_padding
    }

    for n in matrix_sizes:
        print(f"Benchmarking {n}x{n} matrices...")
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        for name, func in algorithms_to_benchmark.items():
            kwargs = {}
            # Pass specific arguments only if needed by the function
            if "Block" in name:
                kwargs['block_size'] = block_size
            elif "Parallel" in name:
                kwargs['num_threads'] = num_threads
            elif "Hybrid" in name:
                kwargs['threshold'] = hybrid_threshold

            # Skip benchmarking the naive algorithm for very large matrices
            if name == "Naive Standard" and n > 256:
                duration = float('inf')
                print(f"  - Skipping {name} for size {n} (too slow)")
            else:
                try:
                    # Run a warm-up execution first
                    time_algorithm(func, A, B, **kwargs)
                    
                    # Run the timed execution
                    duration = time_algorithm(func, A, B, **kwargs)
                    print(f"  - {name:20s}: {duration:.6f} s")
                except Exception as e:
                    print(f"Algorithm {name} failed for size {n}: {e}")
                    duration = float('inf')
            
            results.append({
                "Algorithm": name,
                "Matrix Size": n,
                "Execution Time (s)": duration
            })
            
    return pd.DataFrame(results)