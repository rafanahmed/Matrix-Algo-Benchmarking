# In src/main.py
import pandas as pd
from src.benchmarking import run_benchmark_suite

if __name__ == '__main__':
    # Define benchmark parameters
    MATRIX_SIZES = [16, 32, 64, 128, 256, 512, 1024]
    BLOCK_SIZE = 32
    NUM_THREADS = 8
    HYBRID_THRESHOLD = 64

    print("--- Starting Matrix Multiplication Benchmark Suite ---")
    
    # Run the benchmarks
    results_df = run_benchmark_suite(
        matrix_sizes=MATRIX_SIZES,
        block_size=BLOCK_SIZE,
        num_threads=NUM_THREADS,
        hybrid_threshold=HYBRID_THRESHOLD
    )
    
    print("\n" + "="*50)
    print("           Benchmark Results Summary")
    print("="*50)
    
    # Display the results table nicely formatted
    print(results_df.to_string())

    # Save the results to a CSV file for later analysis/visualization
    results_df.to_csv("results/benchmark_data/latest_benchmark.csv", index=False)
    print("\nBenchmark data saved to 'results/benchmark_data/latest_benchmark.csv'")