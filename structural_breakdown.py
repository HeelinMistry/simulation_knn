import os
from pathlib import Path


def print_project_tree(root_dir, indent="", skip_dirs=None):
    if skip_dirs is None:
        skip_dirs = {".git", "__pycache__", "venv", ".ipynb_checkpoints"}

    root = Path(root_dir)
    items = sorted(list(root.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))

    for i, item in enumerate(items):
        if item.name in skip_dirs:
            continue

        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "

        print(f"{indent}{connector}{item.name}")

        if item.is_dir():
            new_indent = indent + ("    " if is_last else "│   ")
            print_project_tree(item, new_indent, skip_dirs)


if __name__ == "__main__":
    print(f"Project Root: {os.path.basename(os.getcwd())}")
    print_project_tree(os.getcwd())