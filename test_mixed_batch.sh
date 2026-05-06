#!/bin/bash

# Test script for mixed batch training with LMDB

set -e

echo "=== Testing Mixed Batch Training ==="
echo ""


# Run training with mixed batch
echo "Starting training with mixed_batch=True..."
echo ""

dp --pt train test_mptraj/lmdb_mixed_batch.json --skip-neighbor-stat

echo ""
echo "=== Training completed ==="
echo "Check mixed_batch_train.log for details"
