## Building Standard (Naive) Matrix Multiplication

#### The Function Signature & Inputs:

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

#### Validating Inputs & Preparing the Result Matrix:

**Mathematical foundation**:  
The resulting product, matrix $C$, is defined as:  
$AB = C$, where $C$ has dimensions $N \times P$ and  
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


