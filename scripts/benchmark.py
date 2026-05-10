import time
import os
import psutil
import numpy as np
import gc
from src.algorithms.apriori import AprioriBaseline
from src.algorithms.tensor_eclat import TensorEclat
import pandas as pd

NUM_RUNS = 3  # Project requirement: run each experiment >= 3 times and average

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
            if transaction:
                transactions.append(transaction)
    return transactions

def run_single_experiment(algorithm_class, data, min_support, **kwargs):
    """Run a single trial of an algorithm and capture metrics."""
    gc.collect()  # Force garbage collection before measuring
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB
    
    algo = algorithm_class(min_support=min_support, **kwargs)
    
    start_time = time.time()
    frequent_itemsets = algo.fit(data)
    end_time = time.time()
    
    mem_after = process.memory_info().rss / (1024 ** 2)  # MB
    
    exec_time = end_time - start_time
    peak_mem = max(0, mem_after - mem_before)
    total_itemsets = sum(len(v) for v in frequent_itemsets.values())
    candidates = algo.candidate_count
    
    return exec_time, peak_mem, total_itemsets, candidates

def run_experiment_averaged(algorithm_class, data, min_support, num_runs=NUM_RUNS, **kwargs):
    """Run an algorithm NUM_RUNS times and return averaged metrics."""
    # Warm-up run to initialize CUDA context / JIT compilation
    _ = run_single_experiment(algorithm_class, data, min_support, **kwargs)
    
    times, mems, itemset_counts, cand_counts = [], [], [], []
    
    for run in range(num_runs):
        t, m, items, cands = run_single_experiment(algorithm_class, data, min_support, **kwargs)
        times.append(t)
        mems.append(m)
        itemset_counts.append(items)
        cand_counts.append(cands)
    
    return {
        'Execution Time (s)': round(np.mean(times), 4),
        'Time Std Dev (s)': round(np.std(times), 4),
        'Memory Used (MB)': round(np.mean(mems), 2),
        'Frequent Itemsets': int(np.mean(itemset_counts)),
        'Candidates Generated': int(np.mean(cand_counts)),
    }

def main():
    datasets = {
        'Chess': 'data/raw/chess.dat',
        'Connect': 'data/raw/connect.dat',
        'Accidents': 'data/raw/accidents.dat',
    }
    
    supports = [0.6, 0.7, 0.8, 0.9]  # Multiple thresholds for scalability curves
    results = []

    print(f"Starting FIM Benchmarks ({NUM_RUNS} runs per configuration)...")
    
    for ds_name, ds_path in datasets.items():
        print(f"\nLoading {ds_name}...")
        data = load_data(ds_path)
        if not data:
            continue
        print(f"  Loaded {len(data)} transactions.")
            
        for min_sup in supports:
            print(f"  Testing min_sup = {min_sup}")
            
            # 1. Baseline Apriori
            print(f"    Running Apriori Baseline ({NUM_RUNS} runs)...")
            metrics_apr = run_experiment_averaged(AprioriBaseline, data, min_sup)
            metrics_apr.update({'Dataset': ds_name, 'Min Support': min_sup, 'Algorithm': 'Apriori'})
            results.append(metrics_apr)
            print(f"      -> {metrics_apr['Execution Time (s)']}s avg")
            
            # 2. SOTA GPU Tensor Eclat
            print(f"    Running Tensor Eclat ({NUM_RUNS} runs)...")
            metrics_eclat = run_experiment_averaged(TensorEclat, data, min_sup)
            metrics_eclat.update({'Dataset': ds_name, 'Min Support': min_sup, 'Algorithm': 'Tensor Eclat'})
            results.append(metrics_eclat)
            print(f"      -> {metrics_eclat['Execution Time (s)']}s avg")

    df_results = pd.DataFrame(results)
    os.makedirs('data/processed', exist_ok=True)
    df_results.to_csv('data/processed/benchmark_results.csv', index=False)
    print("\nBenchmarking complete! Results saved to data/processed/benchmark_results.csv")
    print(df_results.to_markdown())

if __name__ == '__main__':
    main()
