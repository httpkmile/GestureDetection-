import json
import re

path = r"c:\Users\camil\configgesture\notebooks\webcam_detection_integrated6.ipynb"

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 1 (Imports)
# Cell 1 is nb['cells'][1] (index 1 code cell)
import_cell = nb['cells'][1]
source = import_cell['source']
new_source = []
for line in source:
    if "import cv2" in line:
        new_source.append(line.replace("import cv2", "import cv2 as cv"))
    elif "import joblib" in line:
        new_source.append(line.replace("import joblib", "import joblib as jb"))
    else:
        new_source.append(line)
import_cell['source'] = new_source

# 2. Update all cells: replace "cv2." with "cv." and "joblib." with "jb."
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        modified_source = []
        for line in source:
            # Simple replacements (assuming no strings or complex expressions that should be preserved)
            # Match word boundary for cv2. and joblib.
            line = re.sub(r'\bcv2\.', 'cv.', line)
            line = re.sub(r'\bjoblib\.', 'jb.', line)
            modified_source.append(line)
        cell['source'] = modified_source

# 3. Save
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated with aliases cv and jb.")
