import math
import os
import numpy as np

# ---------------------------------------------------------------------------
#  Import every algorithm we need, no matter whether algorithms.py is in
#  the project root or under src/.
# ---------------------------------------------------------------------------
try:
    from src.algorithms import (
        naive_algorithm,
        optimized_standard_algorithm,
        strassens_algorithm,
        strassen_padding,
        block_multiply_algorithm,
        parallel_naive_algorithm,
        hybrid_strassen_padding,
    )
except ModuleNotFoundError:
    from src.algorithms import (
        naive_algorithm,
        optimized_standard_algorithm,
        strassens_algorithm,
        strassen_padding,
        block_multiply_algorithm,
        parallel_naive_algorithm,
        hybrid_strassen_padding,
    )

# ---------------------------------------------------------------------------
#  Handy helpers
# ---------------------------------------------------------------------------
def _pretty_header(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)

def _verify(reference: np.ndarray, candidate: np.ndarray, name: str) -> None:
    if np.allclose(reference, candidate):
        print(f"  ✓  {name:30s} matches reference")
    else:
        print(f"  ✗  {name:30s} *** DOES NOT MATCH ***")

# ---------------------------------------------------------------------------
#  1) NON-SQUARE MATRIX TEST (all algorithms that accept rectangular input)
# ---------------------------------------------------------------------------
_pretty_header("TEST 1  ·  RECTANGULAR MATRICES")

rows_A, cols_A = 3, 4     # 3×4
rows_B, cols_B = 4, 2     # 4×2
assert cols_A == rows_B

A_ns = np.random.randint(0, 10, (rows_A, cols_A))
B_ns = np.random.randint(0, 10, (rows_B, cols_B))

print("Matrix A (rectangular):\n", A_ns)
print("\nMatrix B (rectangular):\n", B_ns)

ref = optimized_standard_algorithm(A_ns, B_ns)      # baseline

_verify(ref, naive_algorithm(A_ns, B_ns),                 "naive_algorithm")
_verify(ref, block_multiply_algorithm(A_ns, B_ns, 2),     "block_multiply_algorithm")
_verify(ref, parallel_naive_algorithm(A_ns, B_ns, 
                                      num_threads=os.cpu_count()), "parallel_naive_algorithm")
_verify(ref, strassen_padding(A_ns, B_ns),                "strassen_padding")
_verify(ref, hybrid_strassen_padding(A_ns, B_ns),         "hybrid_strassen_padding")

print("\nReference result (A @ B):\n", ref)


# ---------------------------------------------------------------------------
#  2) SQUARE POWER-OF-TWO TEST (required by raw Strassen’s)
# ---------------------------------------------------------------------------
_pretty_header("TEST 2  ·  SQUARE 4×4 MATRICES")

N = 4
A_sq = np.random.randint(0, 10, (N, N))
B_sq = np.random.randint(0, 10, (N, N))

print("Matrix A (4×4):\n", A_sq)
print("\nMatrix B (4×4):\n", B_sq)

ref_sq = optimized_standard_algorithm(A_sq, B_sq)   # baseline

_verify(ref_sq, naive_algorithm(A_sq, B_sq),                    "naive_algorithm")
_verify(ref_sq, strassens_algorithm(A_sq, B_sq),                 "strassen_algorithm")
_verify(ref_sq, block_multiply_algorithm(A_sq, B_sq, 2),        "block_multiply_algorithm")
_verify(ref_sq, parallel_naive_algorithm(A_sq, B_sq, 
                                         num_threads=os.cpu_count()), "parallel_naive_algorithm")
_verify(ref_sq, hybrid_strassen_padding(A_sq, B_sq),            "hybrid_strassen_padding")

print("\nReference result (A @ B):\n", ref_sq)


# ---------------------------------------------------------------------------
#  3) LARGE-SCALE CHECK (optional quick sanity for performance variants)
# ---------------------------------------------------------------------------
_pretty_header("TEST 3  ·  LARGE 256×256 MATRICES  (quick sanity)")

N_large = 256
A_big = np.random.rand(N_large, N_large)
B_big = np.random.rand(N_large, N_large)

# Only test the fast variants to keep run-time reasonable
fast_baseline = optimized_standard_algorithm(A_big, B_big)
_verify(fast_baseline, strassen_padding(A_big, B_big),          "strassen_padding  (256×256)")
_verify(fast_baseline, hybrid_strassen_padding(A_big, B_big),   "hybrid_strassen_padding (256×256)")

print("\nAll tests completed.")