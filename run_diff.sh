#!/usr/bin/env bash
set -euo pipefail

OLD="/aisi/mnt/data_nas/liwentao/devel_workspace/temp_revise/deepmd-kit"
NEW="/aisi/mnt/data_nas/liwentao/devel_workspace/deepmd-kit-lmdb"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# 生成两个目录下的 Python 文件相对路径列表
(
cd "$OLD"
find . -type f -name '*.py' -not -path './.git/*' | sort
) > "$tmpdir/old.lst"

(
cd "$NEW"
find . -type f -name '*.py' \
    -not -path './.git/*' \
    -not -path './deepmd-kit/*' \
    | sort
) > "$tmpdir/new.lst"

echo "=== Only in OLD ==="
comm -23 "$tmpdir/old.lst" "$tmpdir/new.lst" || true

echo
echo "=== Only in NEW ==="
comm -13 "$tmpdir/old.lst" "$tmpdir/new.lst" || true

echo
echo "=== Python file diffs ==="
comm -12 "$tmpdir/old.lst" "$tmpdir/new.lst" | while IFS= read -r file; do
if ! cmp -s "$OLD/$file" "$NEW/$file"; then
    echo
    echo "----- $file -----"
    diff -u "$OLD/$file" "$NEW/$file" || true
fi
done
