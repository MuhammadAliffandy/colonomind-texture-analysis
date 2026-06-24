import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageDraw, ImageFont

def get_placeholder_image(mes_class):
    img = Image.new('RGB', (256, 256), color=(200, 200, 200))
    d = ImageDraw.Draw(img)
    text = f"Insert MES {mes_class}\nRaw Image Here"
    # Basic text centering
    d.text((50, 100), text, fill=(50, 50, 50))
    return np.array(img)

def load_or_create_image(img_path, mes_class):
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((256, 256))
            return np.array(img)
        except:
            return get_placeholder_image(mes_class)
    return get_placeholder_image(mes_class)

def generate_presentation_grids(dataset_name, dl_path, texture_path, labels_path, img_dir, output_dir):
    print(f"\n--- Generating Presentation Grids for {dataset_name} ---")
    dl_features = np.load(dl_path)
    texture_features = np.load(texture_path)
    labels = np.load(labels_path)
    
    # Subsample for faster UMAP visualization
    subsample_idx = np.arange(dl_features.shape[0])
    if len(subsample_idx) > 10000:
        np.random.seed(42)
        subsample_idx = np.random.choice(subsample_idx, 10000, replace=False)
        
    dl_sub = dl_features[subsample_idx]
    texture_sub = texture_features[subsample_idx]
    labels_sub = labels[subsample_idx]
    
    # Scale & UMAP
    print("Computing UMAP embeddings...")
    scaler_dl = StandardScaler()
    dl_sub_scaled = scaler_dl.fit_transform(dl_sub)
    reducer_dl = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    umap_dl = reducer_dl.fit_transform(dl_sub_scaled)
    
    scaler_tex = StandardScaler()
    texture_sub_scaled = scaler_tex.fit_transform(texture_sub)
    reducer_tex = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    umap_tex = reducer_tex.fit_transform(texture_sub_scaled)
    
    unique_labels = np.unique(labels_sub)
    colors = sns.color_palette("husl", len(unique_labels))
    
    for label in unique_labels:
        print(f"Generating grid for MES {label}...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Raw Image
        img_path = os.path.join(img_dir, f"mes{label}.jpg")
        img_arr = load_or_create_image(img_path, label)
        axes[0].imshow(img_arr)
        axes[0].set_title(f"Example Raw Image (MES {label})", fontsize=14)
        axes[0].axis('off')
        
        # 2. UMAP Before
        axes[1].scatter(umap_dl[:, 0], umap_dl[:, 1], color='lightgray', alpha=0.3, s=10) # Background
        mask = (labels_sub == label)
        axes[1].scatter(umap_dl[mask, 0], umap_dl[mask, 1], color=colors[int(label)], label=f'MES {label}', alpha=0.8, s=15)
        axes[1].set_title(f'Raw Deep Learning UMAP (Before)', fontsize=14)
        axes[1].axis('off')
        
        # 3. UMAP After
        axes[2].scatter(umap_tex[:, 0], umap_tex[:, 1], color='lightgray', alpha=0.3, s=10) # Background
        axes[2].scatter(umap_tex[mask, 0], umap_tex[mask, 1], color=colors[int(label)], label=f'MES {label}', alpha=0.8, s=15)
        axes[2].set_title(f'Texture Analysis UMAP (After)', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'{dataset_name.lower()}_presentation_mes{label}.png')
        plt.savefig(out_path, dpi=300)
        plt.close()

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    output_dir = os.path.join(base_dir, 'reports', 'figures')
    img_dir = os.path.join(base_dir, 'data', 'sample_images')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    generate_presentation_grids(
        'LIMUC',
        os.path.join(base_dir, 'data', 'limuc_features', 'limuc_dl_features.npy'),
        os.path.join(base_dir, 'data', 'limuc_features', 'limuc_texture_features.npy'),
        os.path.join(base_dir, 'data', 'limuc_features', 'limuc_labels.npy'),
        img_dir,
        output_dir
    )
