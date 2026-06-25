import matplotlib.pyplot as plt
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

def find_sample_image(limuc_root, mes_label):
    limuc_path = Path(limuc_root)
    if not limuc_path.exists():
        return None
        
    class_mapping = {
        "0": 0, "1": 1, "2": 2, "3": 3,
        "Mayo 0": 0, "Mayo 1": 1, "Mayo 2": 2, "Mayo 3": 3,
        "Mayo_0": 0, "Mayo_1": 1, "Mayo_2": 2, "Mayo_3": 3,
    }
    
    for img_path in limuc_path.rglob("*.*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            if img_path.parent.name in class_mapping:
                if class_mapping[img_path.parent.name] == mes_label:
                    try:
                        img = Image.open(str(img_path)).convert("RGB")
                        return np.array(img.resize((256, 256)))
                    except Exception:
                        continue
    return None

def generate_mes_row(ax_img, ax_bar, mes_label, img_data, features, values, badge_color, circle_indices=[]):
    ax_img.imshow(img_data)
    ax_img.axis('off')
    
    # Draw Badge
    badge_rect = patches.Rectangle((0, -40), 256, 40, facecolor=badge_color, edgecolor='none')
    ax_img.add_patch(badge_rect)
    ax_img.text(128, -20, f"MES {mes_label}", color='white', fontweight='bold', 
                fontsize=16, ha='center', va='center')

    # Draw Bar Chart (Log Scale)
    x_pos = np.arange(len(features))
    # Replace zeros or negative values with small epsilon for log scale
    safe_values = np.where(values <= 0, 1e-6, values)
    ax_bar.bar(x_pos, safe_values, color='#1f77b4', edgecolor='white')
    
    ax_bar.set_yscale('log')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
    ax_bar.set_ylabel("Raw feature value (log scale)", fontsize=8)
    
    ax_bar.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    
    # Circle specific features
    for idx in circle_indices:
        if idx < len(x_pos):
            x = x_pos[idx]
            y = safe_values[idx]
            ellipse = patches.Ellipse((x, y), width=1.5, height=y*0.8, 
                                      edgecolor='red', facecolor='none', lw=2)
            ax_bar.add_patch(ellipse)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "Texture_Analysis_Presentation_Figures")
    os.makedirs(output_dir, exist_ok=True)
    
    limuc_root = "/raid/D13K48009/texture/LIMUC"
    data_dir = os.path.join(base_dir, "data", "limuc_features")
    tex_path = os.path.join(data_dir, "limuc_texture_features.npy")
    lbl_path = os.path.join(data_dir, "limuc_labels.npy")
    
    # Load Real Features if available
    use_real_features = os.path.exists(tex_path) and os.path.exists(lbl_path)
    if use_real_features:
        print("[INFO] Memuat data fitur asli untuk menghitung rata-rata...")
        features_data = np.load(tex_path)
        labels_data = np.load(lbl_path)
    else:
        print("[WARNING] Data fitur tidak ditemukan. Menggunakan dummy values.")
    
    # Features label list (Top 20 for simplicity in visualization, or adapt as needed)
    features = [
        "LL_Mean", "LL_Std", "LL_Var", "LL_Entropy", "LH_Mean", "LH_Std", "LH_Var", "LH_Entropy",
        "HL_Mean", "HL_Std", "HL_Var", "HL_Entropy", "HH_Mean", "HH_Std", "HH_Var", "HH_Entropy",
        "HH_Energy", "GLCM_Contrast", "GLCM_Dissimilarity", "GLCM_Homogeneity"
    ]
    
    # Target 4 rows for MES 0 to 3
    fig = plt.figure(figsize=(14, 12))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(4, 5, figure=fig, width_ratios=[1, 1.5, 1.5, 1.5, 1.5])
    
    colors = ['#2ca02c', '#bcbd22', '#ff7f0e', '#d62728'] # Green, Yellow-ish, Orange, Red
    
    for mes in range(4):
        ax_img = fig.add_subplot(gs[mes, 0])
        ax_bar = fig.add_subplot(gs[mes, 1:])
        
        # 1. Load real image from DGX server
        img_data = find_sample_image(limuc_root, mes)
        if img_data is None:
            img_data = create_placeholder_endoscopy(f"MES {mes}", colors[mes])
            
        # 2. Compute real average values for this MES
        if use_real_features:
            mes_mask = (labels_data == mes)
            if np.sum(mes_mask) > 0:
                # Ambil rata-rata dari 20 fitur pertama (atau disesuaikan dengan array)
                mean_vals = np.mean(np.abs(features_data[mes_mask]), axis=0)
                # Pastikan panjangnya sesuai dengan label fitur
                vals = mean_vals[:len(features)] if len(mean_vals) >= len(features) else np.pad(mean_vals, (0, len(features)-len(mean_vals)))
            else:
                vals = np.random.uniform(10, 100000, size=len(features))
        else:
            vals = np.random.uniform(10, 100000, size=len(features))
            
        # Tentukan fitur yang akan di-highlight (dilingkari merah). Misal fitur index 2 dan 18.
        highlight_idx = [2, 18] # Contoh: LL_Var dan GLCM_Dissimilarity
        
        generate_mes_row(ax_img, ax_bar, str(mes), img_data, features, vals, colors[mes], circle_indices=highlight_idx)
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, "Feature_Importance_Per_MES.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Visualization saved to {out_path}")

if __name__ == '__main__':
    main()
