#!/bin/bash

# Test script for mixed batch training with LMDB

set -e

echo "=== Testing Mixed Batch Training ==="
echo ""

# Check if LMDB data exists
LMDB_PATH="temp/graph_lmdb_demo/out_random_10000/raw_subset.lmdb"
if [ ! -d "$LMDB_PATH" ]; then
    echo "Error: LMDB data not found at $LMDB_PATH"
    echo "Please run the data preparation script first"
    exit 1
fi

# Run training with mixed batch
echo "Starting training with mixed_batch=True..."
echo ""

dp --pt train test_mptraj/lmdb_mixed_batch.json --skip-neighbor-stat

echo ""
echo "=== Training completed ==="
echo "Check mixed_batch_train.log for details"
