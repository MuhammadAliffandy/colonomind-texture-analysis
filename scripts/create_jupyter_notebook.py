import os
import json

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in source.split('\n')]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in source.split('\n')]
    }

def generate_notebook(output_path):
    cells = []
    
    # Cell 1: Intro
    cells.append(create_markdown_cell(
        "# Texture Analysis - Client Presentation\n\n"
        "Gunakan notebook ini untuk me-run analisa tekstur (UMAP StandardScaler, Feature Importance, Rule-Based Thresholds, dan Presentation Grid). \n\n"
        "**PENTING**: Silakan ubah variabel path/direktori di blok bawah ini sesuai letak dataset di server Jupyter Anda."
    ))
    
    # Cell 2: Imports & Variables
    code_imports = """import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# UBAH DIREKTORI INI SESUAI DENGAN SERVER ANDA
# ==========================================
BASE_DIR = "."  # Ganti dengan path root folder proyek Colonomind
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

# Direktori Gambar Mentah (Raw Image Dataset) di Server Jupyter
LIMUC_RAW_DIR = "/Colonoscopy/Dataset/LIMUC"

FEAT_NAMES = [
    "LL_Mean", "LL_Std", "LL_Var", "LL_Ent", 
    "LH_Mean", "LH_Std", "LH_Var", "LH_Ent",
    "HL_Mean", "HL_Std", "HL_Var", "HL_Ent", 
    "HH_Mean", "HH_Std", "HH_Var", "HH_Ent", "HH_Energy",
    "GLCM_Contrast", "GLCM_Dissimilarity", "GLCM_Homogeneity"
]

print("✅ Modul dan Direktori siap.")
"""
    cells.append(create_code_cell(code_imports))
    
    # Cell 3: Function for UMAP and Importance
    code_umap = """def analyze_and_plot_umap(dataset_name, dl_path, texture_path, labels_path):
    print(f"\\n--- Analyzing {dataset_name} ---")
    try:
        dl_features = np.load(dl_path)
        texture_features = np.load(texture_path)
        labels = np.load(labels_path)
    except FileNotFoundError as e:
        print(f"❌ File tidak ditemukan: {e}. Pastikan file .npy ada di direktori yang benar.")
        return None, None, None
        
    subsample_idx = np.arange(dl_features.shape[0])
    if len(subsample_idx) > 10000:
        np.random.seed(42)
        subsample_idx = np.random.choice(subsample_idx, 10000, replace=False)
        
    dl_sub = dl_features[subsample_idx]
    texture_sub = texture_features[subsample_idx]
    labels_sub = labels[subsample_idx]
    
    # StandardScaler
    scaler_dl = StandardScaler()
    dl_sub_scaled = scaler_dl.fit_transform(dl_sub)
    scaler_tex = StandardScaler()
    texture_sub_scaled = scaler_tex.fit_transform(texture_sub)
    
    # UMAP
    print("Computing UMAP embeddings (Ini mungkin memakan waktu 1-2 menit)...")
    reducer_dl = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    umap_dl = reducer_dl.fit_transform(dl_sub_scaled)
    
    reducer_texture = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    umap_texture = reducer_texture.fit_transform(texture_sub_scaled)
    
    # Random Forest for Feature Importance
    print("Training Classification Models...")
    X_train_tex, X_test_tex, y_train, y_test = train_test_split(texture_features, labels, test_size=0.2, random_state=42, stratify=labels)
    rf_tex = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_tex.fit(X_train_tex, y_train)
    
    return (umap_dl, umap_texture, labels_sub), rf_tex, (dl_features, texture_features, labels)

print("Fungsi analisis siap dijalankan.")
"""
    cells.append(create_code_cell(code_umap))
    
    # Cell 4: Execute LIMUC Analysis
    code_execute = """# Mengeksekusi LIMUC
limuc_dl_path = os.path.join(DATA_DIR, "limuc_features", "limuc_dl_features.npy")
limuc_tex_path = os.path.join(DATA_DIR, "limuc_features", "limuc_texture_features.npy")
limuc_lbl_path = os.path.join(DATA_DIR, "limuc_features", "limuc_labels.npy")

limuc_umap_res, limuc_rf, limuc_raw_data = analyze_and_plot_umap("LIMUC", limuc_dl_path, limuc_tex_path, limuc_lbl_path)

if limuc_umap_res:
    umap_dl, umap_texture, labels_sub = limuc_umap_res
    unique_labels = np.unique(labels_sub)
    colors = sns.color_palette("husl", len(unique_labels))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for idx, label in enumerate(unique_labels):
        mask = (labels_sub == label)
        axes[0].scatter(umap_dl[mask, 0], umap_dl[mask, 1], color=colors[idx], label=f'MES {label}', alpha=0.6, s=10)
        axes[1].scatter(umap_texture[mask, 0], umap_texture[mask, 1], color=colors[idx], label=f'MES {label}', alpha=0.6, s=10)
        
    axes[0].set_title('LIMUC UMAP: Raw Deep Learning Features (Before)', fontsize=14)
    axes[1].set_title('LIMUC UMAP: Texture Features (After StandardScaler)', fontsize=14)
    axes[0].legend()
    axes[1].legend()
    plt.show()
    
    # Plot Feature Importance
    importances = limuc_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances for LIMUC (Texture Analysis)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [FEAT_NAMES[i] for i in indices], rotation=45, ha='right')
    plt.xlim([-1, len(importances)])
    plt.show()
"""
    cells.append(create_code_cell(code_execute))
    
    # Cell 5: Rule-based Thresholds
    code_rules = """# Menghitung Thresholds (Aturan)
if limuc_umap_res:
    _, texture_features, labels = limuc_raw_data
    importances = limuc_rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:3] # 3 fitur teratas
    top_features = [FEAT_NAMES[i] for i in top_indices]
    
    df = pd.DataFrame(texture_features, columns=FEAT_NAMES)
    df['Label'] = labels
    
    print("=== LIMUC Rule-Based Thresholds ===")
    print(f"Fitur terpenting: {', '.join(top_features)}\\n")
    
    for label in np.unique(labels):
        print(f"--- MES {label} ---")
        class_df = df[df['Label'] == label]
        for feat_name in top_features:
            q1 = class_df[feat_name].quantile(0.25)
            q3 = class_df[feat_name].quantile(0.75)
            mean_val = class_df[feat_name].mean()
            print(f"{feat_name}: {q1:.4f} sampai {q3:.4f} (Rata-rata: {mean_val:.4f})")
        print("")
"""
    cells.append(create_code_cell(code_rules))

    # Cell 6: Visualisasi Grid (Gambar + UMAP)
    code_grid = """# Membuat Grid Visualisasi untuk Presentasi Klien
def find_sample_image(base_dir, label):
    # Mencari gambar di dalam folder class (misal: 0, 1, 2, 3)
    search_paths = [
        os.path.join(base_dir, "train_and_validation_sets", str(label)),
        os.path.join(base_dir, "patient_based_classified_images", str(label)),
        os.path.join(base_dir, "test_set", str(label))
    ]
    for path in search_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    return os.path.join(path, file)
    return None

def load_or_create_image(dataset_name, mes_class):
    img_path = None
    if dataset_name == "LIMUC":
        img_path = find_sample_image(LIMUC_RAW_DIR, mes_class)
        
    if img_path and os.path.exists(img_path):
        try:
            return np.array(Image.open(img_path).convert('RGB').resize((256, 256)))
        except: pass
    
    # Fallback placeholder jika gambar tidak ditemukan
    img = Image.new('RGB', (256, 256), color=(200, 200, 200))
    d = ImageDraw.Draw(img)
    d.text((50, 100), f"Insert MES {mes_class}\\nRaw Image Here", fill=(50, 50, 50))
    return np.array(img)

if limuc_umap_res:
    umap_dl, umap_texture, labels_sub = limuc_umap_res
    unique_labels = np.unique(labels_sub)
    colors = sns.color_palette("husl", len(unique_labels))
    
    for label in unique_labels:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Raw Image
        img_arr = load_or_create_image("LIMUC", label)
        axes[0].imshow(img_arr)
        axes[0].set_title(f"Example Raw Image (MES {label})")
        axes[0].axis('off')
        
        # 2. UMAP Before
        axes[1].scatter(umap_dl[:, 0], umap_dl[:, 1], color='lightgray', alpha=0.3, s=10)
        mask = (labels_sub == label)
        axes[1].scatter(umap_dl[mask, 0], umap_dl[mask, 1], color=colors[int(label)], label=f'MES {label}', alpha=0.8, s=15)
        axes[1].set_title('Raw Deep Learning UMAP (Before)')
        axes[1].axis('off')
        
        # 3. UMAP After
        axes[2].scatter(umap_texture[:, 0], umap_texture[:, 1], color='lightgray', alpha=0.3, s=10)
        axes[2].scatter(umap_texture[mask, 0], umap_texture[mask, 1], color=colors[int(label)], label=f'MES {label}', alpha=0.8, s=15)
        axes[2].set_title('Texture Analysis UMAP (After)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
"""
    cells.append(create_code_cell(code_grid))

    notebook = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"Jupyter Notebook berhasil dibuat di: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    output_path = os.path.join(base_dir, 'Client_Presentation_Analysis.ipynb')
    generate_notebook(output_path)
