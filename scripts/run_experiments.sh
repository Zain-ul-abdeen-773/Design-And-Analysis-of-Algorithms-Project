#!/bin/bash
# Download datasets and trigger benchmarking

echo "Initializing Environment & Fetching Datasets..."
mkdir -p data/raw data/processed

# Chess dataset
if [ ! -f data/raw/chess.dat ]; then
    echo "Downloading Chess dataset..."
    wget http://fimi.uantwerpen.be/data/chess.dat -O data/raw/chess.dat
fi

# Connect dataset
if [ ! -f data/raw/connect.dat ]; then
    echo "Downloading Connect dataset..."
    wget http://fimi.uantwerpen.be/data/connect.dat -O data/raw/connect.dat
fi

# Accidents dataset
if [ ! -f data/raw/accidents.dat ]; then
    echo "Downloading Accidents dataset..."
    wget http://fimi.uantwerpen.be/data/accidents.dat -O data/raw/accidents.dat
fi

echo "Datasets ready."
echo "Activating benchmark suite..."

echo "Starting automated benchmark..."
python3 scripts/benchmark.py

echo "Generating report figures..."
python3 scripts/generate_plots.py

echo "Launching dashboard (Check your browser)..."
streamlit run scripts/dashboard.py
