## Building Strassen's Algorithm

### Core Concept and Function Setup:

**Mathematical Foundation**:
In 1969, German mathematician Volker Strassen discovered that while the standard (naive) algorithm for multiplying two $2 \times 2$ matrices require 8 multiplications, he could accomplish the same result with only **7 multiplications** at the cost of more additions and subtractions. 

For large matrices, this small saving becomes monumental. The algorithm is **recursive**—a process or program that repeats by calling itself, requiring multiple steps to complete—breaking down a large matrix into smaller and smaller subproblems until they are tiny enough to solve, and then combining the results back together. This process is akin to a mathematical **"Divide and Conquer"** strategy.

**Logic**:
The strategy for an $N \times N$ matrix multiplication $C = AB$ is as follows:
1. **Divide**: If the matrix is the larger than a base case (ex. $1 \times 1$), split matrix $A$ and matrix $B$ into four smaller sub-matrices (quadrants), each of size $(N /  2) \times (N /  2)$. 
   ```ini
A = [ [A11, A12],
      [A21, A22] ]

B = [ [B11, B12],
      [B21, B22] ]
```
1. **Conquer**: Instead of performing 8 recursive multiplications with these sub-matrices, perform only 7, using Strassen's formulas (detailed later in this document).
2. **Combine**: Combine the results of the 7 sub-problems (using more additions and subtractions) to form the quadrants of the final result matrix, $C$. 

To make our initial implementation easier to understand, we will start with the KEY ASSUMPTION: **The input matrices are square and their dimension, N, is a power of 2** (ex. 2, 4, 8, 16, 32...). This ensures that when we divide by 2 repeatedly, we always get integer dimensions and the sub- matrices are always square. We can address non-power-of-two sizes later in this document. 

**Code Implementation**:
We will add a skeleton for our new function `strassens_algorithm()`:

```python
import numpy as np

def strassens_algorithm(A, B):
	"""
	Computes matrix multiplication product C = A * B using Strassen's matrix
	multiplication algorithm.
	
	Args:
		A (np.ndarray): The first square matrix. Its dimension N is a power of 2.
		B (np.ndarray): The second square matrix. Its dimension N is a power of
		2.
	
	Returns:
		np.ndarray: The resulting product matrix.
	"""	
```

**Notes**:
- **Recursion:** Unlike our previous functions that used loops, this function will call _itself_ with smaller inputs. This is the essence of recursion.
- **Function Signature:** The signature is the same as before, accepting two matrices `A` and `B`. The internal logic will be completely different.
- **Assumption:** The docstring clearly states our simplifying assumption. This is good practice so anyone using the function knows its current limitations.


### The Base Case:

**Mathematical Foundation**:
"Divide and conquer" works by breaking a problem down into smaller identical problems. This process must eventually stop. For Strassen's algorithm, the simplest stopping point is when the matrices are reduces to size $1 \times 1$. 

The product of two $1 \times 1$ matrices, $A = (a_{11})$ and $B = (b_{11})$, is simply the scalar product of their singular elements: $$C= (a_{11}\cdot b_{11})$$
**Logic**:
Inside our `strassens_algorithm()` function, the very first thing we will do is check the size of the input matrices. If they are size $1 \times 1$, we'll perform the simple multiplication directly and return the result. This stops the chain of recursive calls. 

**Code Implementation**:
We'll add a check for the dimension `n` at the beginning of the function.
```python
import numpy as np

def strassens_algorithm(A, B):
	"""
	Computes matrix multiplication product C = A * B using Strassen's matrix
	multiplication algorithm.
	
	Args:
		A (np.ndarray): The first square matrix. Its dimension N is a power of 2.
		B (np.ndarray): The second square matrix. Its dimension N is a power of
		2.
	
	Returns:
		np.ndarray: The resulting product matrix.
	"""	
	
	# Get the dimension of the matrix (we assume it's square)
	n = A.shape[0]
	
	# Base case: If the matrices are 1x1, perform simple scalar multiplication
	if n == 1:
		# Return the result as a 1x1 NumPy array to maintain type consistency
		return np.array([[A[0, 0] * B[0, 0]]])
```
**Notes**:
- `n = A.shape[0]`: Since we assume the matrix is square, we only need to get the length of one dimension (the number of rows).
- `if n == 1:`: This is our crucial stopping condition.
- `return np.array([[A[0, 0] * B[0, 0]]])`: It's important that our function _always_ returns a matrix (a 2D NumPy array) to keep the return type consistent. If we just returned `A[0, 0] * B[0, 0]`, we'd be returning a scalar number, which would cause errors when the higher-level recursive calls try to treat it like a matrix. The double brackets `[[...]]` ensure we create a 2D array with one row and one column. 


### Dividing the Matrices:

**Mathematical Foundation**:
In our "Divide and Conquer" strategy, we take our $N \times N$ matrices, $A$ and $B$ and split them each into FOUR $(N /  2) \times (N /  2)$ sub-matrices:   
```ini
A = [ [A11, A12],
      [A21, A22] ]

B = [ [B11, B12],
      [B21, B22] ]
```
- $A_{11}$ is the top-left quadrant of $A$.
- $A_{12}$ is the top-right quadrant of $A$.
- $A_{21}$ is the bottom-left quadrant of $A$.
- $A_{22}$ is the bottom-right quadrant of $A$.
  (And similarly for all of matrix $B$ as well)

**Logic**:
This is where our "else" statement will need to be implemented. We add the "else" part to our `if n == 1:` condition. Inside this block, we will:
- Calculate the midpoint of the dimension `n`.
- Use NumPy's array slicing to create variables for each of the eight sub-matrices (four for $A$, four for $B$).

**Code Implementation**:
We will now add the division logic to the `strassens_algorithm()` function:
```python
import numpy as np

def strassens_algorithm(A, B):
	"""
	Computes matrix multiplication product C = A * B using Strassen's matrix
	multiplication algorithm.
	
	Args:
		A (np.ndarray): The first square matrix. Its dimension N is a power of 2.
		B (np.ndarray): The second square matrix. Its dimension N is a power of
		2.
	
	Returns:
		np.ndarray: The resulting product matrix.
	"""	
	
	# Get the dimension of the matrix (we assume it's square)
	n = A.shape[0]
	
	# Base case: If the matrices are 1x1, perform simple scalar multiplication
	if n == 1:
		# Return the result as a 1x1 NumPy array to maintain type consistency
		return np.array([[A[0, 0] * B[0, 0]]])
	
	# Recursive step: Divide the matrices
	else:
		# 1. Find the midpoint
		mid = n // 2
		
		# 2. Split matrix A into four quadrants
		A11 = A[:mid, :mid]
		A12 = A[:mid, mid:]
		A21 = A[mid:, :mid]
		A22 = A[mid:, mid:]
		
		# 3. Split matrix B into four quadrants
		B11 = B[:mid, :mid]
		B12 = B[:mid, mid:]
		B21 = B[mid:, :mid]
		B22 = B[mid:, mid:]
```
**Notes**:
- `mid = n // 2`: This performs integer division to find the index that splits the matrix in half. Since we're assuming `n` is a power of two, this division will always result in a whole number.
- **NumPy Slicing**: The syntax `array[rows, columns]` is used to select parts of the matrix:
	- `:` on its own means "select all elements along this axis."
	- `:mid` means "from the beginning up to (but not including) the midpoint."
	- `mid:` means "from the midpoint to the very end."
- **Efficiency**: NumPy slicing is highly efficient. It doesn't create copies of the data in memory; it creates "views" that point to the original data, which saves memory and time.


### Strassen's 7 Recursive Formulas:

**Mathematical Foundation**:
Instead of 8 multiplications needed for a standard recursive approach, we calculate 7 special intermediate products, typically denoted as $P_1$ through $P_7$. 
The formulas are:
- $P_{1} = (A_{11}+A_{22})\cdot(B_{11}+B_{22})$
- $P_{2} = (A_{21}+A_{22})\cdot B_{11}$
- $P_{3} = A_{11}\cdot(B_{12}-B_{22})$
- $P_{4} = A_{22}\cdot(B_{21}-B_{11})$
- $P_{5} = (A_{11}+A_{12})\cdot B_{22}$
- $P_{6} = (A_{21}-A_{11})\cdot(B_{11}+B_{12})$
- $P_{7} = (A_{12}-A_{22})\cdot(B_{21}+B_{22})$

**Logic**:
For each formula, we will:
- Perform the matrix additions or subtractions as specified.
- Use the results of those operations as inputs for a **recursive call** to our `strassens_algorithm()` function.

**Code Implementation**:
We'll add the calculations for $P_1$ through $P_7$ to the `else` block, right after we split the matrices. 

```python
import numpy as np

def strassens_algorithm(A, B):
	"""
	Computes matrix multiplication product C = A * B using Strassen's matrix
	multiplication algorithm.
	
	Args:
		A (np.ndarray): The first square matrix. Its dimension N is a power of 2.
		B (np.ndarray): The second square matrix. Its dimension N is a power of
		2.
	
	Returns:
		np.ndarray: The resulting product matrix.
	"""	
	
	# Get the dimension of the matrix (we assume it's square)
	n = A.shape[0]
	
	# Base case: If the matrices are 1x1, perform simple scalar multiplication
	if n == 1:
		# Return the result as a 1x1 NumPy array to maintain type consistency
		return np.array([[A[0, 0] * B[0, 0]]])
	
	# Recursive step: Divide the matrices
	else:
		# 1. Find the midpoint
		mid = n // 2
		
		# 2. Split matrix A into four quadrants
		A11 = A[:mid, :mid]
		A12 = A[:mid, mid:]
		A21 = A[mid:, :mid]
		A22 = A[mid:, mid:]
		
		# 3. Split matrix B into four quadrants
		B11 = B[:mid, :mid]
		B12 = B[:mid, mid:]
		B21 = B[mid:, :mid]
		B22 = B[mid:, mid:]
		
		# 4. Calculate the 7 products using recursive calls
		p1 = strassens_algorithm((A11 + A22), (B11 + B22))
		p2 = strassens_algorithm((A21 + A22), B11)
		p3 = strassens_algorithm(A11, (B12 - B22))
		p4 = strassens_algorithm(A22, (B21 - B11))
		p5 = strassens_algorithm((A11 + A12), B22)
		p6 = strassens_algorithm((A21 - A11), (B11 + B12))
		p7 = strassens_algorithm((A12 - A22), (B21 + B22))
```
**Notes**:
- This section is the core of the algorithm's efficiency gain. Notice we only make **seven** calls to `strassens_algorithm()`, not eight.
- **Intermediate Matrices**: We perform matrix addition and subtraction directly inside the arguments of the recursive calls. For example, `(A11 + A22)` creates a new temporary matrix that is then passed to the `strassens_algorithm()` function.
- **Recursive Nature**: Each of these seven lines is a "Conquer" step that solves a smaller sub-problem of size $(N /  2) \times (N /  2)$. 


### Combining the Results:

**Mathematical Foundation**:
The four quadrants of the result matrix, $C$, are calculated by adding and subtracting the seven intermediate products ($P_1$ through $P_7$) that we just computed. 

The formulas for combining them are:
- $C_{11}=P_{1}+P_{4}-P_{5}+P_{7}$
- $C_{12}=P_{3}+P_{5}$
- $C_{21}=P_{2}+P_{4}$
- $C_{22}=P_{1}-P_{2}+P_{3}+P_{6}$

**Logic**:
Our final steps inside the `else` block are:
1. Calculate each of the four result quadrants (`C11`, `C12`, `C21`, `C22`) using the formulas above.
2. Assemble these four quadrants into a single result matrix, `C`. We can do this by vertically stacking the left-side quadrants (`C11`, `C21`) and the right-side quadrants (`C12`, `C22`), and then horizontally stacking those two resulting columns.
3. Return the final assembled matrix `C`.

**Code Implementation**:
We will now implement our "Combine" logic into our code

```python
import numpy as np

def strassens_algorithm(A, B):
	"""
	Computes matrix multiplication product C = A * B using Strassen's matrix
	multiplication algorithm.
	
	Args:
		A (np.ndarray): The first square matrix. Its dimension N is a power of 2.
		B (np.ndarray): The second square matrix. Its dimension N is a power of
		2.
	
	Returns:
		np.ndarray: The resulting product matrix.
	"""	
	
	# Get the dimension of the matrix (we assume it's square)
	n = A.shape[0]
	
	# Base case: If the matrices are 1x1, perform simple scalar multiplication
	if n == 1:
		# Return the result as a 1x1 NumPy array to maintain type consistency
		return np.array([[A[0, 0] * B[0, 0]]])
	
	# Recursive step: Divide the matrices
	else:
		# 1. Find the midpoint
		mid = n // 2
		
		# 2. Split matrix A into four quadrants
		A11 = A[:mid, :mid]
		A12 = A[:mid, mid:]
		A21 = A[mid:, :mid]
		A22 = A[mid:, mid:]
		
		# 3. Split matrix B into four quadrants
		B11 = B[:mid, :mid]
		B12 = B[:mid, mid:]
		B21 = B[mid:, :mid]
		B22 = B[mid:, mid:]
		
		# 4. Calculate the 7 products using recursive calls
		p1 = strassens_algorithm((A11 + A22), (B11 + B22))
		p2 = strassens_algorithm((A21 + A22), B11)
		p3 = strassens_algorithm(A11, (B12 - B22))
		p4 = strassens_algorithm(A22, (B21 - B11))
		p5 = strassens_algorithm((A11 + A12), B22)
		p6 = strassens_algorithm((A21 - A11), (B11 + B12))
		p7 = strassens_algorithm((A12 - A22), (B21 + B22))
		
		#5. Combine the 7 products to get the quadrants of the result matrix C
		C11 = p1 + p4 - p5 + p7
		C12 = p3 + p5
		C21 = p2 + p4
		C22 = p1 - p2 + p3 + p6
		
		# 6. Assemble the final matrix C from the four quadrants
		# First, create the left and right halves by stacking vertically
		left_half = np.vstack((C11, C21))
		right_half = np.vstack((C12, C22))
		
		# Then, create the full matrix by stacking horizontally
		C = np.hstack((left_half, right_half))
		
		return C
```
**Notes**:
- **Final Assembly**: This is where we reconstruct the full result matrix from its calculated parts.
- `np.vstack((C11, C21))`: NumPy's "vertical stack" function. It takes a tuple of arrays and stacks them on top of each other, creating a taller array.
- `np.hstack((left_half, right_half))`: NumPy's "horizontal stack" function. It takes a tuple of arrays and stacks them side-by-side, creating a wider array.