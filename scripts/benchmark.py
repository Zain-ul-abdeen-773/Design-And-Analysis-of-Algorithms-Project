import time
import os
import psutil
from memory_profiler import memory_usage
from src.algorithms.apriori import AprioriBaseline
from src.algorithms.tensor_eclat import TensorEclat
import pandas as pd

def load_data(filepath, max_transactions=None):
    """Load transactions from FIMI dat format."""
    transactions = []
    if not os.path.exists(filepath):
        print(f"Dataset {filepath} not found.")
        return transactions

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if max_transactions and i >= max_transactions:
                break
            # Split by space and filter out empty strings
            transaction = [int(item) for item in line.strip().split() if item]
            transactions.append(transaction)
    return transactions

def run_experiment(algorithm, data, min_support):
    """Run an algorithm and capture metrics."""
    # Capture start time and memory
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2) # MB
    start_time = time.time()
    
    # Run algorithm
    frequent_itemsets = algorithm.fit(data)
    
    # Capture end time and memory
    end_time = time.time()
    mem_after = process.memory_info().rss / (1024 ** 2) # MB
    
    exec_time = end_time - start_time
    peak_mem = max(0, mem_after - mem_before)  # Approximate peak memory difference
    
    # Calculate total frequent itemsets
    total_itemsets = sum(len(itemsets) for k, itemsets in frequent_itemsets.items())
    candidates = algorithm.candidate_count
    
    return {
        'Execution Time (s)': exec_time,
        'Memory Used (MB)': peak_mem,
        'Frequent Itemsets': total_itemsets,
        'Candidates Generated': candidates
    }

def main():
    datasets = {
        'Chess': 'data/raw/chess.dat',
        'Connect': 'data/raw/connect.dat'
    }
    
    supports = [0.6, 0.7, 0.8] # typical high supports for dense datasets
    results = []

    print("Starting FIM Benchmarks...")
    
    for ds_name, ds_path in datasets.items():
        print(f"\nLoading {ds_name}...")
        data = load_data(ds_path, max_transactions=10000) # Limit for testing
        if not data:
            continue
            
        for min_sup in supports:
            print(f"  Testing min_sup = {min_sup}")
            
            # 1. Baseline Apriori
            print("    Running Apriori Baseline...")
            apriori = AprioriBaseline(min_support=min_sup)
            metrics_apr = run_experiment(apriori, data, min_sup)
            metrics_apr.update({'Dataset': ds_name, 'Min Support': min_sup, 'Algorithm': 'Apriori'})
            results.append(metrics_apr)
            
            # 2. SOTA GPU Tensor Eclat
            print("    Running Tensor Eclat (GPU)...")
            tensor_eclat = TensorEclat(min_support=min_sup)
            metrics_eclat = run_experiment(tensor_eclat, data, min_sup)
            metrics_eclat.update({'Dataset': ds_name, 'Min Support': min_sup, 'Algorithm': 'Tensor Eclat'})
            results.append(metrics_eclat)

    df_results = pd.DataFrame(results)
    os.makedirs('data/processed', exist_ok=True)
    df_results.to_csv('data/processed/benchmark_results.csv', index=False)
    print("\nBenchmarking complete! Results saved to data/processed/benchmark_results.csv")
    print(df_results.to_markdown())

if __name__ == '__main__':
    main()
