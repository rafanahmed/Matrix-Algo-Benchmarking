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

def strassens_algorithm(A, B):
    n = A.shape[0]
    if n == 1:
        return np.array([[A[0, 0] * B[0, 0]]])
    else:
        mid = n // 2
        A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
        B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, :mid], B[mid:, mid:]
        p1 = strassens_algorithm((A11 + A22), (B11 + B22))
        p2 = strassens_algorithm((A21 + A22), B11)
        p3 = strassens_algorithm(A11, (B12 - B22))
        p4 = strassens_algorithm(A22, (B21 - B11))
        p5 = strassens_algorithm((A11 + A12), B22)
        p6 = strassens_algorithm((A21 - A11), (B11 + B12))
        p7 = strassens_algorithm((A12 - A22), (B21 + B22))
        C11 = p1 + p4 - p5 + p7
        C12 = p3 + p5
        C21 = p2 + p4
        C22 = p1 - p2 + p3 + p6
        left_half = np.vstack((C11, C21))
        right_half = np.vstack((C12, C22))
        C = np.hstack((left_half, right_half))
        return C