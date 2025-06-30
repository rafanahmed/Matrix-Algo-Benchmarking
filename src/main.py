import numpy as np
from src.algorithms import naive_algorithm, optimized_standard_algorithm, strassens_algorithm

if __name__ == "__main__":
    # --- Test Case 1: Non-Square Matrices (for Naive & Optimized) ---
    print("--- Testing with Non-Square Matrices ---")
    rows_A, cols_A = 2, 3 # 2x3 Matrix A
    rows_B, cols_B = 3, 2 # 3x2 Matrix B
    assert cols_A == rows_B

    matrix_A_ns = np.random.randint(1, 11, size=(rows_A, cols_A))
    
    matrix_B_ns = np.random.randint(1, 11, size=(rows_B, cols_B))
    
    print("\nMatrix A:\n", matrix_A_ns)
    print("\nMatrix B:\n", matrix_B_ns)

    print("\nExecuting naive_algorithm()...")
    naive_result = naive_algorithm(matrix_A_ns, matrix_B_ns) 
    print("Result from naive_algorithm():\n", naive_result)

    print("\nExecuting optimized_standard_algorithm() wrapper...")
    optimized_result = optimized_standard_algorithm(matrix_A_ns, matrix_B_ns)
    print("Result from optimized_standard_algorithm():\n", optimized_result)

    if np.allclose(naive_result, optimized_result):
        print("\nVerification SUCCESS: The results are identical.")
    else:
        print("\nVerification FAILED: The results are different.")

    print("-" * 40)

    # --- Test Case 2: Square, Power-of-2 Matrices (for all algorithms) ---
    print("\n--- Testing with Square, Power-of-2 Matrices ---")
    N = 4
    matrix_A_sq = np.random.randint(0, 10, (N, N)) # 4x4
    matrix_B_sq = np.random.randint(0, 10, (N, N)) # 4x4

    print("\nMatrix A:\n", matrix_A_sq)
    print("\nMatrix B:\n", matrix_B_sq)

    print("\nExecuting naive_algorithm()...")
    naive_result = naive_algorithm(matrix_A_sq, matrix_B_sq) 
    print("Result from naive_algorithm():\n", naive_result)

    print("\nExecuting optimized_standard_algorithm() wrapper...")
    optimized_result = optimized_standard_algorithm(matrix_A_sq, matrix_B_sq)
    print("Result from optimized_standard_algorithm():\n", optimized_result)

    print("\nExecuting strassens_algorithm()...")
    strassen_result = strassens_algorithm(matrix_A_sq, matrix_B_sq)
    print("Result from strassens_algorithm():\n", strassen_result)

    if np.allclose(naive_result, optimized_result) and np.allclose(naive_result, strassen_result):
        print("\nVerification SUCCESS: The results are identical.")
    else:
        print("\nVerification FAILED: The results are different.")

    print("-" * 40)