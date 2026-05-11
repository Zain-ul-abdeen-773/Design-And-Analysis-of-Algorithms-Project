import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Frequent Itemset Mining (FIM) Benchmark Dashboard", layout="wide")

st.title("Frequent Itemset Mining (FIM) Benchmarking Dashboard 🚀")
st.markdown("Comparing **Classical Apriori** (CPU) against **Tensor Eclat** (GPU) on FIMI Benchmark datasets.")

data_path = 'data/processed/benchmark_results.csv'

if not os.path.exists(data_path):
    st.error(f"Benchmark data not found at {data_path}. Please run `scripts/benchmark.py` first.")
else:
    df = pd.read_csv(data_path)
    
    # Overview metrics
    st.header("1. Overview of Raw Results")
    st.dataframe(df.style.highlight_max(axis=0, subset=['Execution Time (s)']))
    
    datasets = df['Dataset'].unique()
    
    for dataset in datasets:
        st.subheader(f"Dataset Analysis: {dataset}")
        ds_data = df[df['Dataset'] == dataset]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Execution Time vs Minimum Support")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.lineplot(data=ds_data, x='Min Support', y='Execution Time (s)', hue='Algorithm', marker='o', ax=ax)
            ax.set_title(f'Scalability in Time on {dataset}')
            ax.invert_xaxis()
            st.pyplot(fig)
            
        with col2:
            st.write("### Peak Memory Usage")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=ds_data, x='Min Support', y='Memory Used (MB)', hue='Algorithm', ax=ax)
            ax.set_title(f'Memory Footprint on {dataset}')
            st.pyplot(fig)
            
    # Speedup calculations
    st.header("2. Speedup Metrics")
    speed_data = []
    for dataset in datasets:
        ds_data = df[df['Dataset'] == dataset]
        supports = ds_data['Min Support'].unique()
        for sup in supports:
            baseline = ds_data[(ds_data['Min Support'] == sup) & (ds_data['Algorithm'] == 'Apriori')]['Execution Time (s)'].values
            sota = ds_data[(ds_data['Min Support'] == sup) & (ds_data['Algorithm'] == 'Optimized')]['Execution Time (s)'].values
            if len(baseline) > 0 and len(sota) > 0:
                speedup = baseline[0] / sota[0]
                speed_data.append({'Dataset': dataset, 'Min Support': sup, 'Speedup (x)': speedup})
    
    speedup_df = pd.DataFrame(speed_data)
    st.dataframe(speedup_df)
    
    st.markdown("### Speedup Ratio Chart (Baseline / SOTA)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=speedup_df, x='Min Support', y='Speedup (x)', hue='Dataset', ax=ax)
    st.pyplot(fig)
