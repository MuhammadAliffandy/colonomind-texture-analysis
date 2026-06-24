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
        "Gunakan notebook ini untuk me-run analisa tekstur (UMAP StandardScaler, Feature Importance, Rule-Based Thresholds, dan Presentation Grid) untuk dataset **LIMUC** dan **TMC**.\n\n"
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
import warnings
warnings.filterwarnings('ignore')

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
TMC_RAW_DIR = "/Colonoscopy/Dataset/TMC-UCM"

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
    
    # Cell 4: Execute LIMUC & TMC
    code_execute = """def run_analysis_for_dataset(dataset_name, dl_path, tex_path, lbl_path):
    umap_res, rf_model, raw_data = analyze_and_plot_umap(dataset_name, dl_path, tex_path, lbl_path)

    if umap_res:
        umap_dl, umap_texture, labels_sub = umap_res
        unique_labels = np.unique(labels_sub)
        colors = sns.color_palette("husl", len(unique_labels))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for idx, label in enumerate(unique_labels):
            mask = (labels_sub == label)
            axes[0].scatter(umap_dl[mask, 0], umap_dl[mask, 1], color=colors[idx], label=f'MES {label}', alpha=0.6, s=10)
            axes[1].scatter(umap_texture[mask, 0], umap_texture[mask, 1], color=colors[idx], label=f'MES {label}', alpha=0.6, s=10)
            
        axes[0].set_title(f'{dataset_name} UMAP: Raw Deep Learning Features (Before)', fontsize=14)
        axes[1].set_title(f'{dataset_name} UMAP: Texture Features (After StandardScaler)', fontsize=14)
        axes[0].legend()
        axes[1].legend()
        plt.show()
        
        # Plot Feature Importance
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 6))
        plt.title(f"Feature Importances for {dataset_name} (Texture Analysis)")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [FEAT_NAMES[i] for i in indices], rotation=45, ha='right')
        plt.xlim([-1, len(importances)])
        plt.show()
        
        return umap_res, rf_model, raw_data
    return None, None, None

# Eksekusi LIMUC
limuc_res = run_analysis_for_dataset(
    "LIMUC",
    os.path.join(DATA_DIR, "limuc_features", "limuc_dl_features.npy"),
    os.path.join(DATA_DIR, "limuc_features", "limuc_texture_features.npy"),
    os.path.join(DATA_DIR, "limuc_features", "limuc_labels.npy")
)

# Eksekusi TMC
tmc_res = run_analysis_for_dataset(
    "TMC",
    os.path.join(DATA_DIR, "tmc_features", "tmc_dl_features.npy"),
    os.path.join(DATA_DIR, "tmc_features", "tmc_texture_features.npy"),
    os.path.join(DATA_DIR, "tmc_features", "tmc_labels.npy")
)
"""
    cells.append(create_code_cell(code_execute))
    
    # Cell 5: Rule-based Thresholds
    code_rules = """def print_rules(dataset_name, raw_data, rf_model):
    _, texture_features, labels = raw_data
    importances = rf_model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:3] # 3 fitur teratas
    top_features = [FEAT_NAMES[i] for i in top_indices]
    
    df = pd.DataFrame(texture_features, columns=FEAT_NAMES)
    df['Label'] = labels
    
    print(f"=== {dataset_name} Rule-Based Thresholds ===")
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

if limuc_res[0]: print_rules("LIMUC", limuc_res[2], limuc_res[1])
if tmc_res[0]: print_rules("TMC", tmc_res[2], tmc_res[1])
"""
    cells.append(create_code_cell(code_rules))

    # Cell 6: Visualisasi Grid (Gambar + UMAP)
    code_grid = """def find_sample_image(base_dir, dataset_name, label):
    # Logika pencarian folder berdasarkan screenshot server
    if dataset_name == "LIMUC":
        search_paths = [
            os.path.join(base_dir, "train_and_validation_sets", str(label)),
            os.path.join(base_dir, "patient_based_classified_images", str(label)),
            os.path.join(base_dir, "test_set", str(label)),
            os.path.join(base_dir, str(label))
        ]
    else: # TMC-UCM
        search_paths = [
            os.path.join(base_dir, "images", str(label)),
            os.path.join(base_dir, "augment", str(label)),
            os.path.join(base_dir, str(label))
        ]
        
    for path in search_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    return os.path.join(path, file)
    return None

def load_or_create_image(dataset_name, mes_class):
    base_dir = LIMUC_RAW_DIR if dataset_name == "LIMUC" else TMC_RAW_DIR
    img_path = find_sample_image(base_dir, dataset_name, mes_class)
        
    if img_path and os.path.exists(img_path):
        try:
            return np.array(Image.open(img_path).convert('RGB').resize((256, 256)))
        except: pass
    
    # Fallback placeholder jika gambar tidak ditemukan
    img = Image.new('RGB', (256, 256), color=(200, 200, 200))
    d = ImageDraw.Draw(img)
    d.text((50, 100), f"Insert {dataset_name}\\nMES {mes_class} Image Here", fill=(50, 50, 50))
    return np.array(img)

def plot_presentation_grid(dataset_name, umap_res):
    if not umap_res[0]: return
    umap_dl, umap_texture, labels_sub = umap_res[0]
    unique_labels = np.unique(labels_sub)
    colors = sns.color_palette("husl", len(unique_labels))
    
    print(f"\\n--- Visualisasi Grid {dataset_name} ---")
    for label in unique_labels:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Raw Image
        img_arr = load_or_create_image(dataset_name, label)
        axes[0].imshow(img_arr)
        axes[0].set_title(f"{dataset_name} Raw Image (MES {label})")
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

plot_presentation_grid("LIMUC", limuc_res)
plot_presentation_grid("TMC", tmc_res)
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
