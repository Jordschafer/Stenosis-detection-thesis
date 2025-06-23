import os
import ast
import sys
import importlib
import importlib.metadata
from collections import defaultdict
import sysconfig

def find_imports_in_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=file_path)
        except SyntaxError:
            return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                base_name = alias.name.split('.')[0]
                imports.add(base_name)
    return imports

def collect_all_imports(root_dir):
    all_imports = set()
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".py"):
                file_path = os.path.join(dirpath, file)
                all_imports.update(find_imports_in_file(file_path))
    return sorted(all_imports)

def get_package_versions(packages):
    versions = {}
    stdlib_paths = sysconfig.get_paths()["stdlib"]

    for package in packages:
        try:
            # Try to find the installed location of the module
            spec = importlib.util.find_spec(package)
            if spec and spec.origin and not spec.origin.startswith(stdlib_paths):
                # Only include if not from standard lib
                version = importlib.metadata.version(package)
                versions[package] = version
            else:
                continue  # Skip stdlib
        except importlib.metadata.PackageNotFoundError:
            continue  # Skip not-installed
        except Exception:
            continue  # Skip any odd case
    return versions

def main(project_dir):
    print(f" Scanning Python project in: {project_dir}")
    imports = collect_all_imports(project_dir)
    versions = get_package_versions(imports)

    print(f"\n Python version: {sys.version.split()[0]}\n")
    print(" Detected libraries and versions:\n")
    for pkg in sorted(versions):
        print(f"• {pkg} (version {versions[pkg]})")
    with open("requirements_detected.txt", "w") as f:
        f.write(f"Python version: {sys.version.split()[0]}\n\n")
        for pkg in sorted(versions):
            if versions[pkg] != "Not installed":
                f.write(f"{pkg}=={versions[pkg]}\n")
    print("\n Saved to requirements_detected.txt (only installed packages)")
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check all Python libraries used in a project and their versions.")
    parser.add_argument("project_dir", help="Path to the Python project directory")
    args = parser.parse_args()
    main(args.project_dir)
        # Save to file
    

