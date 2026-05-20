#!/bin/bash

# Test script for mixed batch training with LMDB

set -e

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

echo "=== Testing Mixed Batch Training ==="
echo ""

echo "Starting training with mixed_batch=True..."
echo ""

dp --pt train test_mptraj/lmdb_mixed_batch.json --skip-neighbor-stat >mixed_batch_train.log 2>&1

echo ""
echo "=== Training completed ==="
echo "Check mixed_batch_train.log for details"
