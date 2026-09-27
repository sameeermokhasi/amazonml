#!/bin/bash
set -e

echo "================================================="
echo "        FULL PIPELINE EXECUTION STARTED"
echo "================================================="

echo ""
echo "[1/3] Running Blocking Pipeline..."
# We use top_k=300 for maximum recall
python3 src/blocking.py --output candidate_pairs.parquet --top_k 50

echo ""
echo "[2/4] Running Features Pipeline..."
python3 src/features.py --candidates dataset_processed/candidate_pairs.parquet --output dataset_processed/features.parquet

echo ""
echo "[3/4] Running Training Pipeline..."
python3 src/train.py --features dataset_processed/features.parquet

echo ""
echo "[4/4] Running Evaluation Pipeline..."
python3 src/evaluate.py

echo ""
echo "================================================="
echo "        FULL PIPELINE EXECUTION COMPLETED"
echo "================================================="
