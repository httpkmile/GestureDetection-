import json
import os

path = r"c:\Users\camil\configgesture\notebooks\segmentacao_clipseg_v4.ipynb"

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Markdown
cell0 = nb['cells'][0]
cell0['source'] = [
    "# Segmentação de Imagens com CLIPSeg (Nova Versão)\n",
    "\n",
    "Este notebook utiliza o modelo **CIDAS/clipseg-rd64-refined** (CLIPSegForImageSegmentation) para realizar a segmentação de fotos específicas com prompts personalizados para cada uma.\n",
    "\n",
    "**Nota:** Os avisos de \"UNEXPECTED\" (como em `position_ids`) no relatório de carregamento do modelo são normais para este modelo e não afetam o seu funcionamento. Eles ocorrem porque alguns pesos salvos no checkpoint são recalculados dinamicamente pelo modelo durante a inferência."
]

# Find the code cells to modify
# In the original file, image_dir = "imagens" was in nb['cells'][2] 
# (Based on the Line numbers I saw, it was around notebook line 182)

# Cell 2: Parameters and Path
cell2 = nb['cells'][2]
source2 = cell2['source']
new_source2 = []
for line in source2:
    if 'image_dir = "imagens"' in line:
        new_source2.append('image_dir = "../data/imagens"\n')
    else:
        new_source2.append(line)
cell2['source'] = new_source2

# Cell 2: Also contains the segment_image function in some structure? 
# Wait, let me check the cell indices.
# Cell 1 is imports and model load.
# Cell 2 is the function and loop parameters.

premium_function = """def segment_image(image_path, prompts):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    preds = torch.sigmoid(outputs.logits)
    if len(prompts) == 1:
        preds = preds.unsqueeze(0)
    
    n_prompts = len(prompts)
    rows = (n_prompts + 1 + 1) // 2
    cols = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows), dpi=100)
    axes = axes.flatten()
    
    # Imagem Original
    axes[0].imshow(image)
    axes[0].set_title(f"Original: {os.path.basename(image_path)}", fontsize=14, fontweight='bold')
    axes[0].axis("off")
    
    import numpy as np
    # Cores/Cmaps diferentes para cada prompt
    cmaps = ["Reds", "Blues", "Greens", "Purples", "Oranges"]
    
    for i in range(n_prompts):
        ax = axes[i+1]
        # Mostra imagem original no fundo
        ax.imshow(image)
        
        # Processa a máscara
        mask = preds[i].cpu().numpy()
        mask_resized = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, resample=Image.BILINEAR)) / 255.0
        
        # Sobrepõe a máscara com transparência
        overlay = ax.imshow(mask_resized, alpha=0.5, cmap=cmaps[i % len(cmaps)])
        ax.set_title(f"Segmentação: {prompts[i]}", fontsize=14, fontweight='bold')
        ax.axis("off")
        
    # Hide unused axes
    for j in range(n_prompts + 1, len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    plt.show()
"""

# The function in the notebook might be in a separate cell or the same.
# Let's check cell 2's source.
if "def segment_image" in "".join(new_source2):
    # It's in the same cell. We need to replace it.
    # We'll reconstruct the cell source.
    lines = "".join(new_source2).split("def segment_image")[0]
    cell2['source'] = [lines] + [premium_function]

# Now let's save the notebook.
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
