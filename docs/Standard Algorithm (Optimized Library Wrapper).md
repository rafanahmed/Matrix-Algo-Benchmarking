## Building Standard Algorithm (Optimized Library Wrapper) Matrix Multiplication


### Concept and Purpose:

**Mathematical Foundation**:

The objective is to compute the product between two given matrices:

- Let $A$ be a matrix with dimensions $N \times M$ ($N$ rows, $M$ columns).
- Let $B$ be a matrix with dimensions $M \times P$ ($M$ rows, $P$ columns).
  
The resulting product, matrix $C$, is defined as:  
$AB = C$ 
where $C$ has dimensions $N \times P$ and  
$C \in \mathbb{R}^{N \times P}$.

Matrix multiplication is only defined when the number of columns in matrix $A$ matches the number of rows in matrix $B$.  
That is:  
$M_{\text{A columns}} = M_{\text{B rows}}$

Each element in matrix $C$ is computed as:

$c_{ij} = \sum_{k=0}^{M-1} a_{ik} \cdot b_{kj}$

Where:
- $c_{ij}$ is the element in the $i$-th row and $j$-th column of matrix $C$.
- $a_{ik}$ is the element in the $i$-th row and $k$-th column of matrix $A$.
- $b_{kj}$ is the element in the $k$-th row and $j$-th column of matrix $B$.
- $k$ is the summation index, iterating from $1$ to $M$.

This means:
- For each $k$ from $0$ to $M-1$:
- Multiply the $k$-th element of the $i$-th row of matrix $A$ with the $k$-th element of the $j$-th column of matrix $B$.
- Sum those products to get $c_{ij}$.

Visualized:
- ![Naive Matrix Multiplication](./assets/naive-algo.gif)

**Logic**:
- Instead of writing `for` loops, we leverage a pre-existing, highly optimized function provided by the NumPy library. 
	- NumPy has a matrix multiplication function that is not written in pure Python. 
	- The function utilized low-level libraries written in C or Fortran (like BLAD - Basic Linear Algebra Subprograms) that are specifically tuned to get the maximum performance out of the CPU hardware
- The implementation will be a simple Python function that acts as a **wrapper**. It will accept two matrices and pass them directly to NumPy's multiplication operator. 
- The purpose of creating this wrapper function is to have a consistent interface for all our algorithms, which will make our benchmarking code clean and simple later on.

**Code Implementation**:
```python
import numpy as np

def optimized_standard_algorithm(A, B):
	"""
	Computes the matrix product using NumPy's highly optimized implementation.
	This serves as a fast, standard baseline for performance comparison. 
	
	Args:
		A (np.ndarray): The first matrix.
		B (np.ndarray): The second matrix.
		
	Returns: 
		np.ndarray: The resulting product matrix. 
	"""
	# NumPy's '@' operator is a highly optimized matrix multiplication routine.
	# It automatically handles dimension checks for us.
	return A @ B
```
- **Wrapper Function:** `optimized_standard_algorithm` doesn't contain complex logic itself. Its sole job is to call the powerful tool provided by the library.
- **The `@` Operator:** `@` is Python's dedicated operator for matrix multiplication. When used with NumPy arrays, it calls `np.matmul()`, which is the highly optimized function.
- **Performance:** The speed of this function comes from the fact that the actual calculations are not being done by the Python interpreter loop-by-loop. They are performed by pre-compiled code that can take advantage of modern CPU features like cache-aware blocking, SIMD instructions (performing one operation on multiple pieces of data simultaneously), and multi-threading.
- **Built-in Error Checking:** Unlike our naive implementation where we had to manually check dimensions, `np.matmul` (and the `@` operator) has this validation built-in. It will raise a `ValueError` automatically if the matrices are incompatible.