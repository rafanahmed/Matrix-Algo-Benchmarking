import math
import multiprocessing
import concurrent.futures 
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

def strassen_padding(A, B):
    n_a_rows, m_a_cols = A.shape
    n_b_rows, p_b_cols = B.shape
    if m_a_cols != n_b_rows:
        raise ValueError("Incompatible matrix dimensions for multiplication.")
    max_dim = max(n_a_rows, m_a_cols, p_b_cols)
    next_pow_of_2 = int(2 ** math.ceil(math.log2(max_dim)))
    A_padded = np.zeros((next_pow_of_2, next_pow_of_2))
    B_padded = np.zeros((next_pow_of_2, next_pow_of_2))
    A_padded[:n_a_rows, :m_a_cols] = A
    B_padded[:n_b_rows, :p_b_cols] = B
    C_padded = strassens_algorithm(A_padded, B_padded)
    C = C_padded[:n_a_rows, :p_b_cols]
    return C

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
    
def block_multiply_algorithm(A, B, block_size):
    N, M = A.shape
    M, P = B.shape
    C = np.zeros((N,P))
    for ii in range(0, N, block_size):
        for jj in range(0, P, block_size):
            for kk in range(0, M, block_size):
                for i in range(ii, min(ii + block_size, N)):
                    for j in range(jj, min(jj + block_size, P)):
                        for k in range(kk, min(kk + block_size, M)):
                            C[i, j] += A[i, k] * B[k, j]
    return C

def parallel_naive_algorithm(A, B, num_threads: int | None = None):
    N, M = A.shape
    M, P = B.shape
    C = np.zeros((N,P))
    def _worker(start_row, end_row):
        for i in range(start_row, end_row):
            for j in range(P):
                for k in range(M):
                    C[i, j] += A[i, k] * B[k, j]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        chunk_size = math.ceil(N / num_threads)
        tasks = []
        for i in range(0, N, chunk_size):
            start = i
            end = min(i + chunk_size, N)
            task = executor.submit(_worker, start, end)
            tasks.append(task)
        concurrent.futures.wait(tasks)
    return C
   
def hybrid_strassen_padding(A, B, threshold=64):
    n_a_rows, m_a_cols = A.shape
    n_b_rows, p_b_cols = B.shape
    if m_a_cols != n_b_rows:
        raise ValueError("Incompatible matrix dimensions for multiplication.")
    max_dim = max(n_a_rows, m_a_cols, p_b_cols)
    next_pow_of_2 = int(2 ** math.ceil(math.log2(max_dim)))
    A_padded = np.zeros((next_pow_of_2, next_pow_of_2))
    B_padded = np.zeros((next_pow_of_2, next_pow_of_2))
    A_padded[:n_a_rows, :m_a_cols] = A
    B_padded[:n_b_rows, :p_b_cols] = B
    C_padded = hybrid_strassens_algorithm(A_padded, B_padded, threshold)
    C = C_padded[:n_a_rows, :p_b_cols]
    return C

def hybrid_strassens_algorithm(A, B, threshold):
    n = A.shape[0]
    if n <= threshold:
        return strassens_algorithm(A, B)
    else:
        mid = n // 2
        A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
        B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, :mid], B[mid:, mid:]
        p1 = hybrid_strassens_algorithm((A11 + A22), (B11 + B22), threshold)
        p2 = hybrid_strassens_algorithm((A21 + A22), B11, threshold)
        p3 = hybrid_strassens_algorithm(A11, (B12 - B22), threshold)
        p4 = hybrid_strassens_algorithm(A22, (B21 - B11), threshold)
        p5 = hybrid_strassens_algorithm((A11 + A12), B22, threshold)
        p6 = hybrid_strassens_algorithm((A21 - A11), (B11 + B12), threshold)
        p7 = hybrid_strassens_algorithm((A12 - A22), (B21 + B22), threshold)
        C11 = p1 + p4 - p5 + p7
        C12 = p3 + p5
        C21 = p2 + p4
        C22 = p1 - p2 + p3 + p6
        left_half = np.vstack((C11, C21))
        right_half = np.vstack((C12, C22))
        C = np.hstack((left_half, right_half))
        return C