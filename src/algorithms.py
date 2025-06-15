import numpy as np

def naive_algorithm(A, B):
    N, M = A.shape
    M_B, P = B.shape
    if M != M_B:
        raise ValueError(f"Incompatible matrix dimensions: Matrix A has {M} columns, but Matrix B has {M_B} rows.")
    C = np.zeros((N, P))
    for i in range(N):
        for j in range(P):
            for k in range(M):
                C[i, j] += A[i, k] * B[k, j]
    return C

def optimized_standard_algorithm(A, B):
    return A @ B