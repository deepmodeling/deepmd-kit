#!/usr/bin/env python3
"""
Detect affected tests based on changed files and global mapping.
"""
import json
import os
import sys
import subprocess

def debug(msg):
    """Print debug message to stderr so it doesn't mix with output parsing."""
    print(f"DEBUG: {msg}", file=sys.stderr)

def get_changed_files_from_git():
    """Fallback: get changed files directly from git diff."""
    debug("Attempting to get changed files from git diff...")
    
    # Get base SHA from environment (set by workflow)
    base_sha = os.environ.get("COMPARE_SHA")
    if not base_sha:
        debug("COMPARE_SHA not found in environment")
        # Try alternative environment variables
        base_sha = os.environ.get("GITHUB_BASE_SHA")
        if not base_sha:
            # Last resort: use HEAD~1
            base_sha = "HEAD~1"
            debug(f"Using fallback base SHA: {base_sha}")
    
    # Get current SHA
    current_sha = os.environ.get("GITHUB_SHA", "HEAD")
    debug(f"Git diff range: {base_sha}..{current_sha}")
    
    try:
        # Run git diff to get changed files
        result = subprocess.run(
            ["git", "diff", "--name-only", base_sha, current_sha],
            capture_output=True,
            text=True,
            check=True
        )
        
        changed_files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        debug(f"Git diff returned {len(changed_files)} changed files")
        for cf in changed_files:
            debug(f"  - {cf}")
        return changed_files
        
    except subprocess.CalledProcessError as e:
        debug(f"Git diff failed: {e.stderr}")
        return []
    except Exception as e:
        debug(f"Error running git diff: {e}")
        return []

def main():
    # ========== 1. Read changed files (Priority order) ==========
    changed_files = []
    
    # Priority 1: Read from file (most reliable, passed from workflow)
    changed_files_path = os.environ.get("CHANGED_FILES_PATH")
    debug(f"CHANGED_FILES_PATH={changed_files_path}")
    
    if changed_files_path and os.path.exists(changed_files_path):
        debug(f"Reading changed files from file: {changed_files_path}")
        try:
            with open(changed_files_path, "r") as f:
                content = f.read()
                debug(f"Raw file content: {repr(content)}")
                changed_files = [line.strip() for line in content.splitlines() if line.strip()]
        except Exception as e:
            debug(f"Failed to read file: {e}")
    
    # Priority 2: Read from single-line environment variable
    if not changed_files:
        raw = os.environ.get("CHANGED_FILES_SINGLELINE", "")
        debug(f"CHANGED_FILES_SINGLELINE={repr(raw)}")
        if raw:
            changed_files = raw.split()
    
    # Priority 3: Fallback to old CHANGED_FILES variable
    if not changed_files:
        raw = os.environ.get("CHANGED_FILES", "")
        debug(f"CHANGED_FILES={repr(raw)}")
        if raw:
            # Handle both newline and space separated formats
            if "\n" in raw:
                changed_files = [line.strip() for line in raw.splitlines() if line.strip()]
            else:
                changed_files = raw.split()
    
    # Priority 4: Fallback to git diff (NEW - most reliable when env vars fail)
    if not changed_files:
        debug("No changed files found via environment variables, trying git diff...")
        changed_files = get_changed_files_from_git()
    
    debug(f"Parsed changed files ({len(changed_files)}):")
    for cf in changed_files:
        debug(f"  - {cf}")
    
    # If no changed files, trigger full test (safe default)
    if not changed_files:
        debug("No changed files found after all attempts, falling back to full test")
        print("selected_paths=source/tests")
        print("skip_all=false")
        print("need_full_test=true")
        sys.exit(0)
    
    # ========== 2. Load global mapping ==========
    mapping_path = ".global_test_mapping.json"
    debug(f"Loading mapping from: {mapping_path}")
    
    if not os.path.exists(mapping_path):
        debug(f"Mapping file not found, falling back to full test")
        print("selected_paths=source/tests")
        print("skip_all=false")
        print("need_full_test=true")
        sys.exit(0)
    
    try:
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
    except json.JSONDecodeError as e:
        debug(f"Invalid JSON in mapping file: {e}")
        print("selected_paths=source/tests")
        print("skip_all=false")
        print("need_full_test=true")
        sys.exit(0)
    
    # Filter out metadata field
    mapping = {k: v for k, v in mapping.items() if k != "metadata"}
    debug(f"Loaded {len(mapping)} test mappings (excluded metadata)")
    
    # ========== 3. Match changed files to tests ==========
    affected_tests = set()
    new_source_files_without_tests = []
    
    for cf in changed_files:
        debug(f"Processing changed file: {cf}")
        
        # Case A: Changed file is a test file itself
        if cf.startswith("source/tests/"):
            debug(f"Changed file is a test file, adding to affected tests: {cf}")
            affected_tests.add(cf)
            continue
        
        # Case B: Changed file is a workflow/config file - trigger full test
        if cf.startswith(".github/workflows/") or cf.startswith(".github/scripts/"):
            debug(f"Changed workflow/script file {cf}, triggering full test")
            print("selected_paths=source/tests")
            print("skip_all=false")
            print("need_full_test=true")
            sys.exit(0)
        
        # Case C: Changed file is a source file, look for dependent tests
        if cf.startswith("source/deepmd/"):
            found_match = False
            for test_file, meta in mapping.items():
                dependencies = meta.get("dependencies", [])
                if cf in dependencies:
                    debug(f"Found test {test_file} depends on {cf}, adding to affected tests")
                    affected_tests.add(test_file)
                    found_match = True
            
            if not found_match:
                debug(f"Changed source file {cf} has no test coverage, marking for full test")
                new_source_files_without_tests.append(cf)
    
    # ========== 4. Output results ==========
    if new_source_files_without_tests:
        # New source files without test coverage -> full test
        debug(f"New source files without test coverage: {new_source_files_without_tests}")
        print("selected_paths=source/tests")
        print("skip_all=false")
        print("need_full_test=true")
    elif affected_tests:
        # Found affected tests -> run only those
        selected = " ".join(sorted(affected_tests))
        debug(f"Affected tests found: {selected}")
        print(f"selected_paths={selected}")
        print("skip_all=false")
        print("need_full_test=false")
    else:
        # No affected tests and no new source files -> skip tests
        debug("No affected tests found and no new source files")
        print("selected_paths=")
        print("skip_all=true")
        print("need_full_test=false")

if __name__ == "__main__":
    main()