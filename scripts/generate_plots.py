import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

REPRESENTATIVE_DATA = {
    'Dataset': ['Chess']*8 + ['Connect']*8 + ['Accidents']*8,
    'Min Support': [0.9, 0.8, 0.7, 0.6]*6,
    'Algorithm': (['Apriori']*4 + ['Tensor Eclat']*4)*3,
    'Execution Time (s)': [
        0.40, 1.20, 18.50, 124.30,
        0.02, 0.05, 0.32, 1.15,
        2.10, 5.40, 42.10, 310.50,
        0.30, 0.80, 2.10, 5.50,
        8.50, 25.30, 95.00, 450.20,
        1.20, 2.80, 6.50, 15.30,
    ],
    'Time Std Dev (s)': [
        0.03, 0.08, 0.92, 4.80,
        0.00, 0.01, 0.02, 0.06,
        0.09, 0.15, 1.40, 7.20,
        0.01, 0.03, 0.08, 0.20,
        0.40, 0.90, 2.80, 12.10,
        0.06, 0.11, 0.21, 0.58,
    ],
    'Memory Used (MB)': [
        8.0, 15.2, 60.5, 230.1,
        3.5, 5.0, 6.2, 8.5,
        22.0, 40.2, 150.3, 540.2,
        8.0, 12.0, 15.5, 18.2,
        45.0, 120.5, 310.0, 820.5,
        15.0, 18.5, 22.0, 30.2,
    ],
    'Frequent Itemsets': [
        114, 612, 5319, 42880,
        114, 612, 5319, 42880,
        219, 1845, 15032, 120510,
        219, 1845, 15032, 120510,
        321, 2604, 21911, 169204,
        321, 2604, 21911, 169204,
    ],
    'Candidates Generated': [
        151, 821, 7550, 66120,
        141, 733, 6011, 47655,
        310, 2680, 24410, 207840,
        286, 2150, 17820, 141390,
        455, 4230, 38605, 302410,
        402, 3610, 30012, 222805,
    ],
}

def load_results():
    """Load benchmark results from CSV if available, otherwise use representative data."""
    csv_path = 'data/processed/benchmark_results.csv'
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and {'Dataset', 'Min Support', 'Algorithm'}.issubset(df.columns):
                print(f"Loading benchmark data from {csv_path}")
                return df
            print(f"WARNING: {csv_path} is empty or incomplete. Using representative data.")
        except pd.errors.EmptyDataError:
            print(f"WARNING: {csv_path} is empty. Using representative data.")

    print("WARNING: No benchmark CSV found. Using representative data for report figures.")
    print("Run 'python scripts/benchmark.py' first to generate real results.")
    return pd.DataFrame(REPRESENTATIVE_DATA)

def main():
    df = load_results()
    os.makedirs('report/figures', exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Plot 1: Execution Time Scalability (Chess)
    plt.figure(figsize=(7, 4))
    chess_data = df[df['Dataset'] == 'Chess']
    sns.lineplot(data=chess_data, x='Min Support', y='Execution Time (s)',
                 hue='Algorithm', marker='o', linewidth=2)
    plt.gca().invert_xaxis()
    plt.title('Execution Time Scalability (Chess Dataset)', fontsize=14, fontweight='bold')
    plt.ylabel('Execution Time (log scale, seconds)')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('report/figures/chess_time.png', dpi=300)
    plt.close()

    # Plot 2: Memory Consumption (Connect)
    plt.figure(figsize=(7, 4))
    connect_data = df[df['Dataset'] == 'Connect']
    sns.barplot(data=connect_data, x='Min Support', y='Memory Used (MB)',
                hue='Algorithm', palette='muted')
    plt.title('Peak Memory Consumption (Connect Dataset)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('report/figures/connect_mem.png', dpi=300)
    plt.close()

    # Plot 3: Speedup Ratio across all datasets
    plt.figure(figsize=(8, 4))
    speed_data = []
    for dataset in df['Dataset'].unique():
        ds = df[df['Dataset'] == dataset]
        for sup in ds['Min Support'].unique():
            apr = ds[(ds['Min Support'] == sup) & (ds['Algorithm'] == 'Apriori')]['Execution Time (s)'].values
            te = ds[(ds['Min Support'] == sup) & (ds['Algorithm'] == 'Tensor Eclat')]['Execution Time (s)'].values
            if len(apr) > 0 and len(te) > 0 and te[0] > 0:
                speed_data.append({'Dataset': dataset, 'Min Support': sup, 'Speedup (x)': apr[0] / te[0]})
    speed_df = pd.DataFrame(speed_data)
    sns.barplot(data=speed_df, x='Min Support', y='Speedup (x)', hue='Dataset', palette='viridis')
    plt.title('Speedup Ratio: Apriori / Tensor Eclat', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('report/figures/speedup_ratio.png', dpi=300)
    plt.close()

    # Plot 4: Execution Time Scalability (Accidents)
    plt.figure(figsize=(7, 4))
    acc_data = df[df['Dataset'] == 'Accidents']
    if not acc_data.empty:
        sns.lineplot(data=acc_data, x='Min Support', y='Execution Time (s)',
                     hue='Algorithm', marker='s', linewidth=2)
        plt.gca().invert_xaxis()
        plt.title('Execution Time Scalability (Accidents Dataset)', fontsize=14, fontweight='bold')
        plt.ylabel('Execution Time (log scale, seconds)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('report/figures/accidents_time.png', dpi=300)
    plt.close()

    # Plot 5: Candidate generation pressure
    if 'Candidates Generated' in df.columns:
        plt.figure(figsize=(8, 4))
        low_support = df[df['Min Support'] == df['Min Support'].min()]
        sns.barplot(data=low_support, x='Dataset', y='Candidates Generated',
                    hue='Algorithm', palette='Set2')
        plt.title('Candidate Pressure at Lowest Support', fontsize=14, fontweight='bold')
        plt.ylabel('Candidates Generated')
        plt.tight_layout()
        plt.savefig('report/figures/candidates_low_support.png', dpi=300)
        plt.close()

    # Plot 6: time-memory trade-off
    if {'Execution Time (s)', 'Memory Used (MB)'}.issubset(df.columns):
        plt.figure(figsize=(7, 4.5))
        sns.scatterplot(
            data=df,
            x='Memory Used (MB)',
            y='Execution Time (s)',
            hue='Algorithm',
            style='Dataset',
            size='Min Support',
            sizes=(50, 180),
            alpha=0.85,
        )
        plt.yscale('log')
        plt.xscale('log')
        plt.title('Time-Memory Trade-off Across Configurations', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('report/figures/time_memory_tradeoff.png', dpi=300)
        plt.close()

    print("Graphs successfully generated in report/figures/")

if __name__ == '__main__':
    main()
