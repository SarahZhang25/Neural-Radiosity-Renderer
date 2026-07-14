"""Count lines of code in directory"""
from pathlib import Path

def count_py_lines(directory_path="."):
    total_lines = 0
    # Match all .py files recursively
    for path in Path(directory_path).rglob("*.py"):
        if "rigidformer" in str(path):
            continue
        if "renderformer" in str(path) and "train" not in str(path):
            continue
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                lines = sum(1 for _ in f)
                total_lines += lines
                print(f"{path}: {lines} lines")
        except Exception as e:
            print(f"Could not read {path}: {e}")
            
    print(f"\nTotal Python lines of code: {total_lines}")

if __name__ == "__main__":
    count_py_lines()