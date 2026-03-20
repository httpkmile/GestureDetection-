import json

path = r"c:\Users\camil\configgesture\notebooks\segmentacao_clipseg_v4.ipynb"

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Markdown
nb['cells'][0]['source'] = [
    "# Segmentação de Imagens com CLIPSeg (Nova Versão)\n",
    "\n",
    "Este notebook utiliza o modelo **CIDAS/clipseg-rd64-refined** (CLIPSegForImageSegmentation) para realizar a segmentação de fotos específicas com prompts personalizados para cada uma.\n",
    "\n",
    "### Relatório de Carregamento\n",
    "Ao carregar o modelo, você poderá ver um relatório com o status **UNEXPECTED** para `position_ids`. \n",
    "\n",
    "**Nota:** Esses avisos são normais e não representam erros. Eles indicam que os IDs de posição, embora presentes no checkpoint, são recalculados dinamicamente pela arquitetura do CLIPSeg no Transformers. A segmentação funcionará corretamente."
]

# Cell 1: Imports and Model Load
nb['cells'][1]['source'] = [
    "from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation\n",
    "from PIL import Image\n",
    "import torch\n",
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "import numpy as np\n",
    "\n",
    "# Suprime avisos desnecessários de carregamento se desejar\n",
    "import logging\n",
    "from transformers import logging as hf_logging\n",
    "hf_logging.set_verbosity_error()\n",
    "\n",
    "model_name = \"CIDAS/clipseg-rd64-refined\"\n",
    "processor = CLIPSegProcessor.from_pretrained(model_name)\n",
    "model = CLIPSegForImageSegmentation.from_pretrained(model_name)\n",
    "\n",
    "print(f\"Modelo {model_name} carregado com sucesso!\")\n"
]

# Cell 2: Premium segment_image and Configuration
nb['cells'][2]['source'] = [
    "image_dir = \"../data/imagens\"\n",
    "\n",
    "images_and_prompts = {\n",
    "    \"passaro.jpg\": [\"bird\", \"branch\", \"eye\"],\n",
    "    \"caoegato.jpg\": [\"cat\", \"dog\", \"floor\"],\n",
    "    \"pizza.jpg\": [\"pizza slice\", \"tomato\", \"cheese\"]\n",
    "}\n",
    "\n",
    "def segment_image(image_path, prompts):\n",
    "    \"\"\"\n",
    "    Realiza a segmentação da imagem com base nos prompts fornecidos e exibe um overlay premium.\n",
    "    \"\"\"\n",
    "    image = Image.open(image_path).convert(\"RGB\")\n",
    "    inputs = processor(text=prompts, images=[image] * len(prompts), padding=\"max_length\", return_tensors=\"pt\")\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        outputs = model(**inputs)\n",
    "    \n",
    "    preds = torch.sigmoid(outputs.logits)\n",
    "    if len(prompts) == 1:\n",
    "        preds = preds.unsqueeze(0)\n",
    "    \n",
    "    n_prompts = len(prompts)\n",
    "    # Layout: 1 imagem original + N segmentações em um grid de 2 colunas\n",
    "    rows = (n_prompts + 1 + 1) // 2\n",
    "    cols = 2\n",
    "    \n",
    "    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows), dpi=100)\n",
    "    axes = axes.flatten()\n",
    "    \n",
    "    # Exibe a Imagem Original\n",
    "    axes[0].imshow(image)\n",
    "    axes[0].set_title(f\"Original: {os.path.basename(image_path)}\", fontsize=14, fontweight='bold', pad=15)\n",
    "    axes[0].axis(\"off\")\n",
    "    \n",
    "    # Paleta de cores para os overlays\n",
    "    cmaps = [\"Reds\", \"Blues\", \"Greens\", \"Purples\", \"Oranges\", \"YlOrBr\"]\n",
    "    \n",
    "    for i in range(n_prompts):\n",
    "        ax = axes[i+1]\n",
    "        # Fundo com a imagem original\n",
    "        ax.imshow(image)\n",
    "        \n",
    "        # Redimensiona a máscara para o tamanho da imagem original\n",
    "        mask = preds[i].cpu().numpy()\n",
    "        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, resample=Image.BILINEAR)\n",
    "        mask_resized = np.array(mask_pil) / 255.0\n",
    "        \n",
    "        # Sobrepõe a máscara colorida com transparência\n",
    "        ax.imshow(mask_resized, alpha=0.5, cmap=cmaps[i % len(cmaps)])\n",
    "        ax.set_title(f\"Segmentação: {prompts[i]}\", fontsize=14, fontweight='bold', pad=15)\n",
    "        ax.axis(\"off\")\n",
    "        \n",
    "    # Remove placeholders não utilizados no grid\n",
    "    for j in range(n_prompts + 1, len(axes)):\n",
    "        axes[j].axis(\"off\")\n",
    "        \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n"
]

# Save with standard formatting
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated with premium features and path fixes.")
