## Building the Parallel Naive Algorithm

### Concept and Purpose:

**The Problem: Idle CPU Cores**
The standard naive algorithm is **sequential**—it executes one instruction at a time on a single CPU core. Since modern processors have multiple cores, a sequential program leaves most of the computer's processing power unused.

**The Solution: Parallelism**
The solution is to divide the main task into smaller, independent sub-tasks and assign each sub-task to a different CPU core to be executed concurrently. For matrix multiplication $C = AB$, the calculation of each row of the result matrix `C` is an independent task. For instance, the calculations for row 5 of `C` do not depend on the results from row 6.

**Logic**:
1.  **Divide the Work**: Split the `N` rows of the output matrix `C` into several chunks.
2.  **Assign Workers**: Create a pool of "worker threads" using Python's `concurrent.futures` library.
3.  **Execute in Parallel**: Assign each worker thread a different chunk of rows to calculate.
4.  **Combine Results**: Since each thread writes its results to a different part of a shared final matrix `C`, the results are automatically combined when all threads are finished.

### Dividing and Executing the Work:

**Mathematical Foundation**:
The mathematical operation is still the standard $C_{ij} = \sum_{k} A_{ik} B_{kj}$. The innovation is in the execution strategy, not the arithmetic.

**Logic**:
1.  **Define a Worker Function**: A nested function, `_worker`, is created to perform the naive multiplication for a specific range of rows (e.g., from `start_row` to `end_row`).
2.  **Use a Thread Pool**: A `ThreadPoolExecutor` manages the creation and execution of threads.
3.  **Submit Tasks**: The main function divides the total rows `N` into chunks and "submits" a task to the pool for each chunk, telling a thread to run the `_worker` function on that specific chunk.

**Code Implementation**:
```python
import numpy as np
import math
import concurrent.futures

def parallel_naive_algorithm(A, B, num_threads=4):
    """
    Computes matrix product C = A * B using a parallelized naive algorithm.

    Args:
        A (np.ndarray): The first matrix.
        B (np.ndarray): The second matrix.
        num_threads (int): The number of worker threads to use.

    Returns:
        np.ndarray: The resulting product matrix.
    """
    N, M = A.shape
    _, P = B.shape
    C = np.zeros((N, P))

    def _worker(start_row, end_row):
        """Worker function that computes a specific range of rows in C."""
        for i in range(start_row, end_row):
            for j in range(P):
                for k in range(M):
                    C[i, j] += A[i, k] * B[k, j]

    # Use a ThreadPoolExecutor to manage worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Divide the N rows into chunks for each thread
        chunk_size = math.ceil(N / num_threads)
        tasks = []
        for i in range(0, N, chunk_size):
            start = i
            end = min(i + chunk_size, N)
            # Assign a chunk of rows to a worker thread
            task = executor.submit(_worker, start, end)
            tasks.append(task)
        
        # Wait for all threads to complete
        concurrent.futures.wait(tasks)
        
    return C
```

Notes:

- Global Interpreter Lock (GIL): In standard Python (CPython), the GIL prevents multiple threads from executing Python bytecode simultaneously. This means for CPU-bound tasks like this, ThreadPoolExecutor may not yield a significant speedup and can even be slower due to thread management overhead. It's included here as an educational example of a common parallelization pattern. True CPU-bound parallelism in Python is typically achieved with the multiprocessing module.