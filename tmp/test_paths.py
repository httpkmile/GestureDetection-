import os

notebook_dir = r"c:\Users\camil\configgesture\notebooks"
project_root = r"c:\Users\camil\configgesture"

print(f"CWD: {os.getcwd()}")

possibilities = [
    "imagens",
    "data/imagens",
    "../data/imagens",
    "assets"
]

for p in possibilities:
    full = os.path.join(notebook_dir, p)
    exists = os.path.exists(full)
    print(f"Path '{p}' from notebook_dir exists: {exists} ({full})")
