## Building Standard (Naive) Matrix Multiplication

### The Function Signature & Inputs:

**Mathematical foundation**:  
The objective is to compute the product between two given matrices:

- Let $A$ be a matrix with dimensions $N \times M$ ($N$ rows, $M$ columns).
- Let $B$ be a matrix with dimensions $M \times P$ ($M$ rows, $P$ columns).



**Logic**:
- Define a Python function that contains our Naive Algorithm where the function needs to accept two matrices as inputs (our arguments). 
- Use NumPy arrays to represent those matrices, given that they are standard for numerical operations in Python. 


**Code Implementation**:
```python
import numpy as np

def naive_algorithm(A, B):
    """
    Computes matrix product C = A * B using the naive matrix
    multiplication algorithm.

    Args:
        A (np.ndarray): The first matrix of shape (N, M).
        B (np.ndarray): The second matrix of shape (M, P).

    Returns:
        np.ndarray: The resulting product matrix of shape (N, P).
    """
    # Rest of code...
```

- `import numpy as np` : Line of code used to access the NumPy library in order to have access to the `ndarray` object, which is an efficient way to store and manipulate matrices. For conventional coding practices, we assign the object with the alias `np`.
- `def naive_algorithm(A, B):` : Line of code that defines our Naive Algorithm function. `A` and `B` are the parameters that will hold the two input matrices when the function is called.
- The text within triple quotes `"""..."""` provides an explanation on what the function does, what arguments (`Args`) it expects, and what it returns (`Returns`).


---

### Validating Inputs & Preparing the Result Matrix:

**Mathematical foundation**:  
The resulting product, matrix $C$, is defined as:  
$AB = C$ 
where $C$ has dimensions $N \times P$ and  
$C \in \mathbb{R}^{N \times P}$.

Matrix multiplication is only defined when the number of columns in matrix $A$ matches the number of rows in matrix $B$.  
That is:  
$M_{\text{A columns}} = M_{\text{B rows}}$



**Logic**:
1. Get & Verify Dimensions:
	- We'll programmatically get the shapes of `A` and `B`. We will then compare the inner dimension (`M`) to ensure they match. If they do not match, the operation is impossible, so we stop the function and alert the user with an error. 
2. Create Result Matrix:
	- Once the inputs are confirmed to be valid, the final matrix `C` will have a shape of `N x P`. We need to create this matrix and fill it with zeros. This will act as a sort of container where we will add our calculated values. 



**Code Implementation**:
```python
import numpy as np

def naive_algorithm(A, B):
    """
    Computes matrix product C = A * B using the naive matrix
    multiplication algorithm.

    Args:
        A (np.ndarray): The first matrix of shape (N, M).
        B (np.ndarray): The second matrix of shape (M, P).

    Returns:
        np.ndarray: The resulting product matrix of shape (N, P).
    """

    #1. Get the dimensions of the input matrices:
    N, M = A.shape
    M_B, P = B.shape # Get shape of B seperately for the check

    #2. Check if number of columns matrix A = number of rows matrix B:
    if M != M_B:
	raise ValueError(f"Incompatible matrix dimensions: Matrix A has {M}
	columns, but Matrix B has {M_B} rows.")

    #3. If the check passes, create the result matrix C, initialized with zeros.
    #   It will have N rows (from A) and P columns (from B)
    C = np.zeros((N, P))
	
    # Rest of code...
```

- `A.shape` : Property of a NumPy array that returns a tuple containing its dimensions. For a 2D matrix, it returns `(number_of_rows, number_of_columns)`.
  ```python
  import numpy as np

  A = np.array([[1, 2, 3],
                [4, 5, 6]])
  print(A.shape)  # Output: (2, 3)

- `N, M = A.shape`: This is a Python feature called "tuple unpacking." It conveniently assigns the first element of the `shape` tuple to `N` and the second to `M`.
- `raise ValueError(...)`: Standard Python way to handle cases where a function receives an argument of the right type but an inappropriate value. Error message is an "f-string", which makes it easy to include the actual mismatched dimensions.
- `np.zeros((N, P))`: This NumPy function creates a new array of a given shape—in this case, `(N, P)`—and fills it entirely with `0.0`. We need this so we can start accumulating our sums in the next step. The double parentheses `((N, P))` are because the `shape` is passed as a single tuple argument.
  ```python
  N, P = 2, 3
  C = np.zeros((N,P))
  print(C)
  # Output:
  #[[0. 0. 0.]
  # [0. 0. 0.]]


---

### The Core Logic - The Three Nested Loops

**Mathematical Foundation**:
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

In order to implement the formula, we need to program a process that calculates the sum for *every single element*  in our result matrix `C`. There are 3 levels of iteration:
1. Outer loop (`i`):
	- We need to iterate through each **row** of our result matrix `C`. This corresponds to the index `i` from $0$ to $N-1$. 
2. Middle Loop (`j`): 
	- For each row `i`, we need to iterate through each **column** of `C`. This corresponds to the index `j` from $0$ to $P−1$.
3. Innermost Loop (`k`): 
	- We've targeted a specific cell `C[i, j]`. We now need to compute the summation. This involves iterating along the shared dimension `M`. 
	- This corresponds to the index `k` from $0$ to $M−1$. Inside this loop, we multiply the corresponding elements `A[i, k]` and `B[k, j]` and add the result to our total for `C[i, j]`.

This three-level loop structure will be incorporated using three nested `for` loops. 


**Code Implementation**:

We will add the loops to our function, right before the `return` statement. This completes the algorithm.

```python
import numpy as np

def naive_algorithm(A, B):
    """
    Computes matrix product C = A * B using the naive matrix
    multiplication algorithm.

    Args:
        A (np.ndarray): The first matrix of shape (N, M).
        B (np.ndarray): The second matrix of shape (M, P).

    Returns:
        np.ndarray: The resulting product matrix of shape (N, P).
    """
    
    #1. Get the dimensions of the input matrices:
    N, M = A.shape
    M_B, P = B.shape # Get shape of B seperately for the check

	#2. Check if number of columns matrix A = number of rows matrix B:
	if M != M_B:
		raise ValueError(f"Incompatible matrix dimensions: Matrix A has {M}
		columns, but Matrix B has {M_B} rows.")

	#3. If the check passes, create the result matrix C, initialized with zeros.
	#   It will have N rows (from A) and P columns (from B)
	C = np.zeros((N, P))


	#4. Iterate through each ROW of the result matrix C (and matrix A):
	for i in range(N):
		#5. For each row, iterate through each COLUMN of C (and matrix B):
		for j in range(P):
			#6. For each cell C[i, j], compute the dot product of row i 
			#   from matrix A and column j from matrix B
			for k in range(M): # M is the shared dimension
				C[i,j] += A[i,k] * B[k,j]

	#7. Return the completed result matrix
	return C
```
- `for i in range(N):`: This loop iterates through the row indices. `i` will go from `0`, `1`, `2`, ..., up to `N-1`.
- `for j in range(P):`: This loop iterates through the column indices. `j` will go from `0`, `1`, `2`, ..., up to `P-1`.
- `for k in range(M):`: This loop performs the summation. It iterates through the elements of the chosen row from `A` and column from `B`. `k` goes from `0`, `1`, `2`, ..., up to `M-1`.
- `C[i, j] += A[i, k] * B[k, j]`: This is the most critical line.
    - `A[i, k]`: Accesses the element in row `i`, column `k` of matrix `A`.
    - `B[k, j]`: Accesses the element in row `k`, column `j` of matrix `B`.
    - Notice how `i` and `j` stay fixed while the inner loop runs, allowing `k` to move along the row of `A` and down the column of `B`.
    - The `+=` operator is shorthand for `C[i, j] = C[i, j] + ...`. Since `C[i, j]` started at `0`, this line accumulates the sum of the products, exactly matching the mathematical formula.
