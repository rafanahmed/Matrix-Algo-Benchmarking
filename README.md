# Matrix Multiplication Algorithm Explorer & Benchmarker

This repository explores the performance and implementation of various matrix multiplication algorithms. The primary goal was to benchmark a range of algorithms, from the classic textbook method to more advanced recursive and parallel approaches, to understand the practical trade-offs between theoretical complexity and real-world performance.

The project includes from-scratch implementations, correctness tests using `pytest`, and a benchmarking suite to measure runtime performance across different matrix sizes.

## How to Set Up and Run This Project

### 1. Prerequisites
- Python 3.8+
- Git

### 2. Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/matrix-algo-visualizer.git](https://github.com/your-username/matrix-algo-visualizer.git)
    cd matrix-algo-visualizer
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python3 -m venv venv

    # Activate it (macOS/Linux)
    source venv/bin/activate
    # Or on Windows
    # .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Run Correctness Tests
To verify that all implemented algorithms produce correct results, run `pytest` from the root directory:
```bash
pytest
````

You should see all tests pass, confirming the implementations are valid.

### 4\. Run Benchmarks

To run the performance benchmark suite, execute the `main.py` script. The script will test the algorithms on various matrix sizes and print a summary table to the console.

```bash
python -m src.main
```

The results will also be saved to `results/benchmark_data/latest_benchmark.csv`.

-----

## Algorithms Implemented

This project successfully implemented and benchmarked six distinct approaches to matrix multiplication.

#### 1\. Standard (Naive) Algorithm

  * **Description**: The fundamental, textbook implementation using three nested loops.
  * **Complexity**: $O(N^3)$
  * **Role**: Serves as the educational baseline and is used to verify the correctness of other, more complex algorithms.

#### 2\. Standard Algorithm (Optimized Library)

  * **Description**: A wrapper for NumPy's `matmul` function (`@` operator), which calls down to highly optimized, low-level BLAS/LAPACK libraries written in C or Fortran.
  * **Complexity**: Technically $O(N^3)$, but with a vastly smaller constant factor due to hardware-specific optimizations.
  * **Role**: Represents the real-world, high-performance baseline that practical applications rely on.

#### 3\. Strassen's Algorithm (with Padding)

  * **Description**: The classic "fast" algorithm that uses a recursive divide-and-conquer strategy to reduce the number of required multiplications from 8 to 7 for each $2 \\times 2$ sub-problem. This implementation includes padding to handle matrices of any size.
  * **Complexity**: Approx. $O(N^{2.807})$
  * **Role**: Demonstrates the concept of reducing asymptotic complexity.

#### 4\. Block Matrix Multiplication (Tiling)

  * **Description**: An iterative algorithm that partitions matrices into smaller square blocks (tiles). It performs the multiplication on these blocks, improving performance by optimizing for the CPU cache (improving data locality).
  * **Complexity**: $O(N^3)$
  * **Role**: Shows how performance can be improved by considering hardware architecture, even without changing the asymptotic complexity.

#### 5\. Parallel Naive Algorithm

  * **Description**: An implementation that attempts to speed up the naive algorithm by splitting the work of calculating rows across multiple threads using Python's `concurrent.futures.ThreadPoolExecutor`.
  * **Complexity**: $O(N^3)$
  * **Role**: Serves as an educational example of a common parallelization pattern.

#### 6\. Hybrid Strassen's Algorithm

  * **Description**: The most advanced implementation in this project. It combines Strassen's recursive approach for large matrices with a fast, iterative algorithm (NumPy's) for sub-problems that are smaller than a specified `threshold`.
  * **Complexity**: Approx. $O(N^{2.807})$
  * **Role**: Represents how "fast" algorithms are often implemented in practice to balance theoretical advantage with real-world overhead.

-----

## Benchmark Analysis: Theoretical Complexity vs. Real-World Performance

The benchmarking results from this project highlight several crucial and sometimes counter-intuitive lessons about algorithm performance.

### Key Findings & Explanations

#### 1\. NumPy is Untouchable

The most immediate observation is that `NumPy Optimized` is orders of magnitude faster than any other implementation.

  * **Why?** NumPy doesn't run in pure Python. It calls highly optimized, pre-compiled C/Fortran libraries (like BLAS) that are specifically tuned for the underlying CPU architecture. They utilize hardware features like SIMD instructions to perform operations on multiple numbers at once, something pure Python cannot do. This demonstrates that the constant factor in Big O notation can be overwhelmingly significant in practice.

#### 2\. Strassen's Overhead is Real

For small to medium matrix sizes (e.g., up to 64x64 or 128x128 in our tests), both the `Strassen (Padded)` and `Hybrid Strassen` algorithms were significantly *slower* than the `Naive Standard` algorithm.

  * **Why?** Strassen's algorithm has a high "overhead." The process of recursively creating 7 sub-problems, performing 18 matrix additions/subtractions, and reassembling the final matrix takes a considerable amount of time. The theoretical savings from doing 7 multiplications instead of 8 only begin to outweigh this overhead when the matrices become large enough that the cost of multiplications truly dominates the runtime. This is why the **Hybrid Strassen's Algorithm** is so important in practice—it mitigates this issue by switching to a faster iterative method for smaller matrices.

#### 3\. The "Parallel" Naive Algorithm Wasn't Faster

The benchmark showed that the `Parallel Naive` implementation was often slower than its single-threaded counterpart.

  * **Why?** This is a classic Python lesson about the **Global Interpreter Lock (GIL)**. The GIL is a mechanism in the standard Python interpreter that prevents multiple threads from executing Python bytecode at the exact same time. For CPU-bound tasks like our numerical loops, creating and managing threads adds overhead without providing true parallel execution. Therefore, the algorithm does not achieve the expected speedup. True CPU-parallelism in Python is typically achieved with the `multiprocessing` module, which uses separate processes to bypass the GIL.

### Conclusion

This project successfully demonstrates the crucial difference between **asymptotic complexity** and **practical performance**. While an algorithm like Strassen's is theoretically superior, factors like algorithmic overhead, hardware-specific optimizations in libraries like NumPy, and language-specific limitations like Python's GIL dictate real-world efficiency. The fastest algorithm in theory is not always the fastest in practice, especially for smaller problem sizes.

