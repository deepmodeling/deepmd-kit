#!/usr/bin/env python3

import json
import sys
import os
import ast
import time

def extract_dependencies_and_functions(test_file):
    """Extract dependencies and test functions from the test file"""
    dependencies = []
    test_functions = []
    
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        if len(content) > 1024000:  # 1MB restriction
            print(f"Skipping large file: {test_file}", file=sys.stderr)
            return None
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            # test files
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                test_functions.append(node.name)
            
            # collect: import deepmd.*
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("deepmd"):
                        module_path = alias.name.replace(".", "/") + ".py"
                        dep = "source/" + module_path
                        dependencies.append(dep)
            
            # collect: from deepmd.xxx import yyy
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("deepmd"):
                base = node.module.replace(".", "/")
                if node.names:
                    for alias in node.names:
                        if alias.name == "*":
                            dep = f"source/{base}.py"
                        else:
                            dep = f"source/{base}/{alias.name}.py"
                        
                        if not os.path.exists(dep):
                            dep = f"source/{base}.py"
                        
                        dependencies.append(dep)
    except SyntaxError:
        pass
    except Exception as e:
        print(f"Error parsing {test_file}: {e}", file=sys.stderr)
    
    return {
        "dependencies": sorted(set(dependencies)),
        "test_functions": sorted(set(test_functions))
    }

def main():
    # Check if the test file list exists
    if not os.path.exists("test_files.txt"):
        print("Error: test_files.txt not found!", file=sys.stderr)
        sys.exit(1)
    
    # read test file list 
    with open("test_files.txt", "r") as f:
        test_files = [line.strip() for line in f if line.strip()]
    
    mapping = {}
    processed = 0
    total = len(test_files)
    start_time = time.time()
    
    for test_file in test_files:
        processed += 1
        if processed % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {processed}/{total} files ({elapsed:.1f}s)", file=sys.stderr)
        
        if not os.path.exists(test_file):
            continue
        
        result = extract_dependencies_and_functions(test_file)
        if result:
            mapping[test_file] = result
    
    # Clear empty entries
    mapping = {k: v for k, v in mapping.items() 
               if v["dependencies"] or v["test_functions"]}
    
    # write global_test_mapping
    with open(".global_test_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    
    elapsed = time.time() - start_time
    print(f" Built global mapping with {len(mapping)} test files in {elapsed:.1f}s", file=sys.stderr)

if __name__ == "__main__":
    main()