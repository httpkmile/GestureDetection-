import json
import os

path = r"c:\Users\camil\configgesture\notebooks\coletar_dados_maos5.ipynb"

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The first code cell is at index 1 (index 0 is markdown)
code_cell = nb['cells'][1]
source = code_cell['source']

# Check if joblib is already there
if not any("import joblib" in line for line in source):
    # Find a good place to insert it (after existing imports)
    insertion_point = 0
    for i, line in enumerate(source):
        if line.strip().startswith("import "):
            insertion_point = i + 1
    
    source.insert(insertion_point, "import joblib\n")
    code_cell['source'] = source

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("joblib added to imports.")
else:
    print("joblib already present.")
