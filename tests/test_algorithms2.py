import numpy as np
import pytest

from src.algorithms import (
    naive_algorithm,
    optimized_standard_algorithm,
    strassens_algorithm,
    strassen_padding,
    block_multiply_algorithm,
    parallel_naive_algorithm,
    hybrid_strassen_padding,
)

def test_rectangular():
    A = np.random.randint(0, 10, (3, 4))
    B = np.random.randint(0, 10, (4, 2))
    reference = optimized_standard_algorithm(A, B)
    assert np.allclose(reference, naive_algorithm(A, B))
    assert np.allclose(reference, block_multiply_algorithm(A, B, 2))
    assert np.allclose(reference, parallel_naive_algorithm(A,B))
    assert np.allclose(reference, strassen_padding(A,B))
    assert np.allclose(reference, hybrid_strassen_padding(A,B))

def test_square_power_of_two():
    N = 4
    A = np.random.randint(0, 10, (N, N))
    B = np.random.randint(0, 10, (N, N))
    reference = optimized_standard_algorithm(A, B)
    assert np.allclose(reference, strassens_algorithm(A, B))

def test_square_non_power_of_two():
    N = 6
    A = np.random.randint(0, 10, (N, N))
    B = np.random.randint(0, 10, (N, N))
    reference = optimized_standard_algorithm(A, B)
    assert np.allclose(reference, strassen_padding(A, B))
    assert np.allclose(reference, hybrid_strassen_padding(A, B))