import nbformat as nbf
import os

def main():
    base_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind-Texture-analysis"
    nb_path = os.path.join(base_dir, "Client_Presentation_Analysis.ipynb")
    
    with open(nb_path, 'r') as f:
        nb = nbf.read(f, as_version=4)
        
    setup_source = """import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont
import matplotlib.patches as patches
from pathlib import Path
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
LIMUC_RAW_DIR = "/home/ubuntu/Colonoscopy/Dataset/LIMUC"
TMC_RAW_DIR = "/home/ubuntu/Colonoscopy/Dataset/TMC-UCM"

FEAT_NAMES = [
    "LL_Mean", "LL_Std", "LL_Var", "LL_Ent", 
    "LH_Mean", "LH_Std", "LH_Var", "LH_Ent",
    "HL_Mean", "HL_Std", "HL_Var", "HL_Ent", 
    "HH_Mean", "HH_Std", "HH_Var", "HH_Ent", "HH_Energy",
    "GLCM_Contrast", "GLCM_Dissimilarity", "GLCM_Homogeneity"
]

# --- Tambahan Konfigurasi Ekstraksi 3 Kanal ---
channels = ['Green', 'Lab-a', 'Lab-b']
feature_names_3ch = []
for ch in channels:
    for f in FEAT_NAMES: 
        feature_names_3ch.append(f"{ch}_{f}")

print("✅ Modul dan Direktori siap.")

# Load Data
dataset_type = "limuc" # Ganti 'tmc' untuk uji coba TMC
tex_path = os.path.join(DATA_DIR, f"{dataset_type}_features", f"{dataset_type}_texture_features.npy")
lbl_path = os.path.join(DATA_DIR, f"{dataset_type}_features", f"{dataset_type}_labels.npy")
dl_path  = os.path.join(DATA_DIR, f"{dataset_type}_features", f"{dataset_type}_dl_features.npy")

print("=== Loading Features ===")
if os.path.exists(tex_path) and os.path.exists(lbl_path):
    features_data = np.load(tex_path)
    labels_data = np.load(lbl_path)
    dl_features = np.load(dl_path) if os.path.exists(dl_path) else None
    print(f"Loaded {len(features_data)} images. Features per image: {features_data.shape[1]}")
else:
    print("[WARNING] Data fitur tidak ditemukan! Menghasilkan data dummy untuk demonstrasi pipeline.")
    np.random.seed(42)
    features_data = np.random.randn(1000, len(feature_names_3ch)) 
    labels_data = np.random.randint(0, 4, 1000)
    dl_features = np.random.randn(1000, 128)"""

    # Ganti isi cell index 1 (cell kode pertama)
    if len(nb.cells) > 1 and nb.cells[1].cell_type == 'code':
        nb.cells[1].source = setup_source
        
    with open(nb_path, 'w') as f:
        nbf.write(nb, f)
        
    print("Setup cell updated!")

if __name__ == "__main__":
    main()
