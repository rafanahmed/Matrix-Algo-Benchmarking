import numpy as np
from src.algorithms import naive_algorithm, optimized_standard_algorithm

if __name__ == "__main__":
    print("--- Running Test Case for Naive Algorithm Multiplication ---")

    matrix_A = np.array([[1, 2, 3], 
                         [4, 5, 6]])
    
    matrix_B = np.array([[7, 8], 
                         [9, 10], 
                         [11, 12]])
    
    print("\nMatrix A:\n", matrix_A)
    print("\nMatrix B:\n", matrix_B)

    print("\nExecuting naive_algorithm()...")
    result = naive_algorithm(matrix_A, matrix_B) 
    print("Result from naive_algorithm():\n", result)

    print("\nExecuting optimized_standard_algorithm() wrapper...")
    optimized_result = optimized_standard_algorithm(matrix_A, matrix_B)
    print("Result from optimized_standard_algorithm():\n", optimized_result)


    if np.allclose(result, optimized_result):
        print("\nVerification SUCCESS: The results are identical.")
    else:
        print("\nVerification FAILED: The results are different.")