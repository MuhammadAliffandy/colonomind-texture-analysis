import nbformat as nbf
import os

def main():
    base_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind-Texture-analysis"
    nb_path = os.path.join(base_dir, "Colab_Feature_Importance_Visualization.ipynb")
    
    nb = nbf.v4.new_notebook()
    
    # Cell 1: Colab Setup (Markdown)
    cell_1_md = nbf.v4.new_markdown_cell("""# 🌟 Feature Importance per MES (Google Colab Version)
Gunakan notebook ini untuk menjalankan visualisasi fitur per MES jika server DGX sedang down.
Pastikan untuk menjalankan cell di bawah ini secara berurutan.""")
    
    # Cell 2: Colab Setup (Code)
    cell_2_code = nbf.v4.new_code_cell("""# 1. Install dependencies
!pip install pyradiomics
!pip install matplotlib numpy Pillow

# 2. Clone repository (opsional jika dijalankan langsung dari Google Drive yang sudah ada reponya)
import os
if not os.path.exists('Colonomind-Texture-analysis'):
    !git clone https://github.com/MuhammadAliffandy/colonomind-texture-analysis.git
    
# Masuk ke direktori
%cd colonomind-texture-analysis""")

    # Cell 3: Markdown
    cell_3_md = nbf.v4.new_markdown_cell("""## Jalankan Script Visualisasi
Jika Anda memiliki data asli (`limuc_texture_features.npy` dan `limuc_labels.npy`) di folder `data/limuc_features/`, script akan menggunakannya. Jika tidak, ia akan menghasilkan visualisasi menggunakan dummy data untuk sementara waktu.""")

    # Cell 4: Python Script
    cell_4_code = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from PIL import Image, ImageDraw
from pathlib import Path

def create_placeholder_endoscopy(text_label, color):
    img = Image.new('RGB', (256, 256), color=(220, 150, 150))
    d = ImageDraw.Draw(img)
    for i in range(10):
        d.arc([(20+i*10, 30+i*5), (200-i*5, 220+i*5)], start=0, end=180, fill=(200, 100, 100), width=3)
    return np.array(img)

def generate_mes_row(ax_img, ax_bar, mes_label, img_data, features, values, badge_color, circle_indices=[]):
    ax_img.imshow(img_data)
    ax_img.axis('off')
    
    badge_rect = patches.Rectangle((0, -40), 256, 40, facecolor=badge_color, edgecolor='none')
    ax_img.add_patch(badge_rect)
    ax_img.text(128, -20, f"MES {mes_label}", color='white', fontweight='bold', 
                fontsize=16, ha='center', va='center')

    x_pos = np.arange(len(features))
    safe_values = np.where(values <= 0, 1e-6, values)
    ax_bar.bar(x_pos, safe_values, color='#1f77b4', edgecolor='white')
    
    ax_bar.set_yscale('log')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
    ax_bar.set_ylabel("Raw feature value (log scale)", fontsize=8)
    
    ax_bar.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    
    for idx in circle_indices:
        if idx < len(x_pos):
            x = x_pos[idx]
            y = safe_values[idx]
            ellipse = patches.Ellipse((x, y), width=1.5, height=y*0.8, 
                                      edgecolor='red', facecolor='none', lw=2)
            ax_bar.add_patch(ellipse)

# Setup path lokal di Colab
output_dir = "Texture_Analysis_Presentation_Figures"
os.makedirs(output_dir, exist_ok=True)

data_dir = os.path.join("data", "limuc_features")
tex_path = os.path.join(data_dir, "limuc_texture_features.npy")
lbl_path = os.path.join(data_dir, "limuc_labels.npy")

use_real_features = os.path.exists(tex_path) and os.path.exists(lbl_path)
if use_real_features:
    print("[INFO] Memuat data fitur asli...")
    features_data = np.load(tex_path)
    labels_data = np.load(lbl_path)
else:
    print("[WARNING] Data fitur tidak ditemukan. Menggunakan dummy values (Silakan upload .npy ke folder data/limuc_features).")

features = [
    "LL_Mean", "LL_Std", "LL_Var", "LL_Entropy", "LH_Mean", "LH_Std", "LH_Var", "LH_Entropy",
    "HL_Mean", "HL_Std", "HL_Var", "HL_Entropy", "HH_Mean", "HH_Std", "HH_Var", "HH_Entropy",
    "HH_Energy", "GLCM_Contrast", "GLCM_Dissimilarity", "GLCM_Homogeneity"
]

fig = plt.figure(figsize=(14, 12))
from matplotlib.gridspec import GridSpec
gs = GridSpec(4, 5, figure=fig, width_ratios=[1, 1.5, 1.5, 1.5, 1.5])

colors = ['#2ca02c', '#bcbd22', '#ff7f0e', '#d62728']

for mes in range(4):
    ax_img = fig.add_subplot(gs[mes, 0])
    ax_bar = fig.add_subplot(gs[mes, 1:])
    
    # Gunakan placeholder untuk gambar (karena dataset asli puluhan GB tidak ada di Colab)
    img_data = create_placeholder_endoscopy(f"MES {mes}", colors[mes])
        
    if use_real_features:
        mes_mask = (labels_data == mes)
        if np.sum(mes_mask) > 0:
            mean_vals = np.mean(np.abs(features_data[mes_mask]), axis=0)
            vals = mean_vals[:len(features)] if len(mean_vals) >= len(features) else np.pad(mean_vals, (0, len(features)-len(mean_vals)))
        else:
            vals = np.random.uniform(10, 100000, size=len(features))
    else:
        vals = np.random.uniform(10, 100000, size=len(features))
        
    highlight_idx = [2, 18] 
    generate_mes_row(ax_img, ax_bar, str(mes), img_data, features, vals, colors[mes], circle_indices=highlight_idx)
    
plt.tight_layout()
out_path = os.path.join(output_dir, "Feature_Importance_Per_MES.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[INFO] Visualization saved to {out_path}")
plt.show()""")

    nb['cells'] = [cell_1_md, cell_2_code, cell_3_md, cell_4_code]
    
    with open(nb_path, 'w') as f:
        nbf.write(nb, f)
        
    print(f"Jupyter Notebook generated at: {nb_path}")

if __name__ == "__main__":
    main()
