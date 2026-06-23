#!/usr/bin/env python3
import subprocess
import sys
import os
import sqlite3

def get_changed_files():
    base = os.environ.get("GITHUB_PR_BASE_SHA", "")
    head = os.environ.get("GITHUB_PR_HEAD_SHA", "")
    if base and head:
        cmd = f"git diff --name-only {base}..{head}"
    else:
        cmd = "git diff --name-only HEAD && git diff --cached --name-only"
    try:
        res = subprocess.check_output(cmd, shell=True, text=True)
        return [l.strip() for l in res.strip().splitlines() if l.strip()]
    except Exception:
        return []

def has_python_changes(files):
    return any(f.endswith(".py") for f in files)

def has_failed_tests():
    """
    检查 testmon 数据库中是否存在失败的测试记录
    testmon 的 failed 字段：1 = failed
    """
    db = ".testmondata"
    if not os.path.exists(db):
        return False
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM node WHERE failed = 1")
        count = cur.fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False

def main():
    changed = get_changed_files()
    db_exists = os.path.exists(".testmondata")

    # 情况 1：没有 .testmondata → 必须跑（种草）
    if not db_exists:
        print("testmon DB not found, cannot skip.")
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("skip_all=false\n")
        return

    # 情况 2：有 Python 改动 → 不能 skip
    if has_python_changes(changed):
        print("Python files changed, cannot skip.")
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("skip_all=false\n")
        return

    # 情况 3：无代码改动，但有失败记录 → 不能 skip（这正是你想要的）
    if has_failed_tests():
        print("Previous failures detected by testmon, cannot skip.")
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("skip_all=false\n")
        return

    # 情况 4：一切安好 → 安全 skip ✅
    print("No changes and no previous failures. Safe to skip.")
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write("skip_all=true\n")

if __name__ == "__main__":
    main()