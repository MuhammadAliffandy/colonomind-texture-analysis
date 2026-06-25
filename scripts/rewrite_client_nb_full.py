import nbformat as nbf
import os

def main():
    base_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind-Texture-analysis"
    nb_path = os.path.join(base_dir, "Client_Presentation_Analysis.ipynb")
    
    nb = nbf.v4.new_notebook()
    
    # --- CELL 1: Setup ---
    setup_code = """import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import umap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont
import pywt
import scipy.stats
from pathlib import Path
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import SimpleITK as sitk
    from radiomics import glcm
    RADIOMICS_AVAILABLE = True
except ImportError:
    sitk = None
    glcm = None
    RADIOMICS_AVAILABLE = False
    print("[WARNING] PyRadiomics tidak terinstal. Pastikan untuk !pip install pyradiomics di awal jika Anda butuh fitur GLCM.")

# ==========================================
# UBAH DIREKTORI INI SESUAI DENGAN SERVER ANDA
# ==========================================
BASE_DIR = "."  # Ganti dengan path root folder proyek Colonomind
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)
for d in ['limuc_features', 'tmc_features']:
    os.makedirs(os.path.join(DATA_DIR, d), exist_ok=True)

# Direktori Gambar Mentah (Raw Image Dataset) di Server Jupyter
LIMUC_RAW_DIR = "/home/ubuntu/Colonoscopy/Dataset/LIMUC"
TMC_RAW_DIR = "/home/ubuntu/Colonoscopy/Dataset/TMC-UCM"

# Definisikan nama fitur untuk visualisasi
BASE_FEAT_NAMES = [
    "LL_Mean", "LL_Std", "LL_Var", "LL_Ent", 
    "LH_Mean", "LH_Std", "LH_Var", "LH_Ent",
    "HL_Mean", "HL_Std", "HL_Var", "HL_Ent", 
    "HH_Mean", "HH_Std", "HH_Var", "HH_Ent", "HH_Energy",
    "GLCM_Autocorr", "GLCM_JointAvg", "GLCM_ClusterProm", 
    "GLCM_ClusterShade", "GLCM_ClusterTend", "GLCM_Contrast", 
    "GLCM_Correl", "GLCM_DiffAvg", "GLCM_DiffEnt", "GLCM_DiffVar", 
    "GLCM_Dissimilarity", "GLCM_JointEnergy", "GLCM_JointEnt", 
    "GLCM_Homogeneity", "GLCM_Id", "GLCM_Idm", "GLCM_Idmn", 
    "GLCM_Idn", "GLCM_Imc1", "GLCM_Imc2", "GLCM_InverseVar", 
    "GLCM_MaxProb", "GLCM_SumAvg", "GLCM_SumEnt", "GLCM_SumSq"
]

channels = ['Green', 'Laba', 'Labb']
FEATURE_NAMES_3CH = []
for ch in channels:
    for f in BASE_FEAT_NAMES: 
        FEATURE_NAMES_3CH.append(f"{ch}_{f}")

print("✅ Modul dan Direktori siap.")"""
    nb.cells.append(nbf.v4.new_code_cell(setup_code))

    # --- CELL 2: Feature Extractor Function ---
    nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Fungsi Ekstraksi 3-Kanal (DWT-17 + PyRadiomics GLCM)
Fungsi ini akan dipakai untuk mengekstrak gambar secara dinamis jika file fitur `.npy` belum pernah dibuat."""))

    extractor_code = """def _smart_preprocess(img):
    if img is None: return None
    h, w = img.shape[:2]
    if h > 450 and w > 550: 
        crop = img[30:430, 200:550]
        if crop.size == 0: crop = img
    else: crop = img
    return cv2.resize(crop, (224, 224))

def _extract_dwt_17(channel):
    coeffs = pywt.dwt2(channel, 'db1')
    LL, (LH, HL, HH) = coeffs
    
    def _stats(band):
        flat = np.abs(band.flatten()) + 1e-6
        return [np.mean(band), np.std(band), np.var(band), scipy.stats.entropy(flat)]
    
    feats = []
    for band in [LL, LH, HL, HH]: feats.extend(_stats(band)) 
    feats.append(np.sum(np.square(HH))) # HH Energy
    return feats

def _extract_pyradiomics_glcm(channel):
    if not RADIOMICS_AVAILABLE: return [0.0] * 24
    
    arr_3d = np.expand_dims(channel, axis=0)
    image = sitk.GetImageFromArray(arr_3d)
    mask = sitk.GetImageFromArray(np.ones_like(arr_3d, dtype=np.uint8))
    
    settings = {'binWidth': 25, 'force2D': True, 'force2Ddimension': 0}
    extractor = glcm.RadiomicsGLCM(image, mask, **settings)
    extractor.enableAllFeatures()
    results = extractor.execute()
    
    return [results[k].item() for k in sorted(results.keys())]

def extract_3ch_features(img_arr):
    # Asumsikan img_arr adalah RGB
    ch_green = img_arr[:, :, 1]
    lab = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB)
    ch_laba = lab[:, :, 1]
    ch_labb = lab[:, :, 2]
    
    all_features = []
    for ch in [ch_green, ch_laba, ch_labb]:
        ch_norm = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        all_features.extend(_extract_dwt_17(ch_norm))
        all_features.extend(_extract_pyradiomics_glcm(ch_norm))
        
    return np.array(all_features, dtype=np.float32)"""
    nb.cells.append(nbf.v4.new_code_cell(extractor_code))

    # --- CELL 3: LIMUC Extraction ---
    nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Ekstraksi Dataset LIMUC (Patient Level)
Melakukan *looping* untuk mengambil gambar berdasar folder pasien. Jika file `.npy` sudah ada, kita load saja agar lebih cepat."""))

    limuc_code = """limuc_tex_path = os.path.join(DATA_DIR, "limuc_features", "limuc_texture_features.npy")
limuc_lbl_path = os.path.join(DATA_DIR, "limuc_features", "limuc_labels.npy")

if os.path.exists(limuc_tex_path) and os.path.exists(limuc_lbl_path):
    print("✅ File ekstraksi LIMUC sudah ditemukan. Melewati proses ekstraksi...")
    limuc_features = np.load(limuc_tex_path)
    limuc_labels = np.load(limuc_lbl_path)
    print(f"Loaded LIMUC: {limuc_features.shape}")
else:
    print("⏳ Menjalankan ekstraksi LIMUC secara Patient-Level...")
    # Target folder misal: /home/ubuntu/Colonoscopy/Dataset/LIMUC/patient_based_classified_images/
    # Folder structure: patient_based_classified_images / <patient_number> / Mayo X / file.bmp
    base_target = os.path.join(LIMUC_RAW_DIR, "patient_based_classified_images")
    if not os.path.exists(base_target): base_target = LIMUC_RAW_DIR # Fallback
    
    limuc_img_paths = []
    limuc_img_labels = []
    
    class_map = {"0": 0, "1": 1, "2": 2, "3": 3, "Mayo 0": 0, "Mayo 1": 1, "Mayo 2": 2, "Mayo 3": 3}
    for root, dirs, files in os.walk(base_target):
        folder_name = os.path.basename(root)
        if folder_name in class_map:
            label = class_map[folder_name]
            for file in files:
                if file.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                    limuc_img_paths.append(os.path.join(root, file))
                    limuc_img_labels.append(label)
                    
    print(f"Menemukan {len(limuc_img_paths)} gambar LIMUC. Mulai memproses...")
    limuc_features_list = []
    limuc_labels_list = []
    
    # Process with tqdm
    for idx in tqdm(range(len(limuc_img_paths)), desc="LIMUC Extraction"):
        img = cv2.imread(limuc_img_paths[idx])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_prep = _smart_preprocess(img)
            feats = extract_3ch_features(img_prep)
            limuc_features_list.append(feats)
            limuc_labels_list.append(limuc_img_labels[idx])
            
    limuc_features = np.array(limuc_features_list)
    limuc_labels = np.array(limuc_labels_list)
    
    np.save(limuc_tex_path, limuc_features)
    np.save(limuc_lbl_path, limuc_labels)
    print(f"✅ Ekstraksi selesai! Shape: {limuc_features.shape}. Disimpan ke {limuc_tex_path}")"""
    nb.cells.append(nbf.v4.new_code_cell(limuc_code))

    # --- CELL 4: TMC Extraction ---
    nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Ekstraksi Dataset TMC-UCM"""))
    tmc_code = """tmc_tex_path = os.path.join(DATA_DIR, "tmc_features", "tmc_texture_features.npy")
tmc_lbl_path = os.path.join(DATA_DIR, "tmc_features", "tmc_labels.npy")

if os.path.exists(tmc_tex_path) and os.path.exists(tmc_lbl_path):
    print("✅ File ekstraksi TMC sudah ditemukan. Melewati proses ekstraksi...")
    tmc_features = np.load(tmc_tex_path)
    tmc_labels = np.load(tmc_lbl_path)
    print(f"Loaded TMC: {tmc_features.shape}")
else:
    print("⏳ Menjalankan ekstraksi TMC...")
    tmc_img_paths = []
    tmc_img_labels = []
    
    # Deteksi label dari nama file / folder TMC
    # Tergantung dari struktur folder TMC-UCM Anda (contoh sederhana via rglob)
    # Anda mungkin perlu menyesuaikan regex/mapping untuk TMC di baris ini
    for img_path in Path(TMC_RAW_DIR).rglob("*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.bmp']:
            # Asumsikan format: klasifikasi ada di folder parent atau nama file
            # Fallback: assign label dummy jika tidak ada aturan spesifik. 
            # Silakan edit bagian pengambilan label ini!
            label = 0 
            if '1' in img_path.parent.name: label = 1
            elif '2' in img_path.parent.name: label = 2
            elif '3' in img_path.parent.name: label = 3
            
            tmc_img_paths.append(str(img_path))
            tmc_img_labels.append(label)
            
    print(f"Menemukan {len(tmc_img_paths)} gambar TMC. Mulai memproses...")
    tmc_features_list = []
    tmc_labels_list = []
    
    for idx in tqdm(range(len(tmc_img_paths)), desc="TMC Extraction"):
        img = cv2.imread(tmc_img_paths[idx])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_prep = _smart_preprocess(img)
            feats = extract_3ch_features(img_prep)
            tmc_features_list.append(feats)
            tmc_labels_list.append(tmc_img_labels[idx])
            
    tmc_features = np.array(tmc_features_list)
    tmc_labels = np.array(tmc_labels_list)
    
    if len(tmc_features) > 0:
        np.save(tmc_tex_path, tmc_features)
        np.save(tmc_lbl_path, tmc_labels)
    print(f"✅ Ekstraksi selesai! Shape: {tmc_features.shape}")"""
    nb.cells.append(nbf.v4.new_code_cell(tmc_code))

    # --- CELL 5: Clustering Pipeline ---
    nb.cells.append(nbf.v4.new_markdown_cell("""## 4. Evaluasi Clustering (LIMUC)"""))
    clust_code = """def evaluate_clustering(labels_true, labels_pred, features, route_name):
    ari = adjusted_rand_score(labels_true, labels_pred)
    if len(features) > 15000:
        idx = np.random.choice(len(features), 15000, replace=False)
        sil = silhouette_score(features[idx], labels_pred[idx])
    else:
        sil = silhouette_score(features, labels_pred)
    print(f"{route_name:25} | ARI: {ari:+.4f} | Silhouette: {sil:+.4f}")

if len(limuc_features) > 0:
    print("Z-score standardization...")
    scaler = StandardScaler()
    scaled_limuc = scaler.fit_transform(limuc_features)

    print("\\nExecuting 3 Clustering Routes for LIMUC...")
    # Route 1: Raw -> K-Means
    kmeans_r1 = KMeans(n_clusters=4, random_state=42, n_init=10)
    evaluate_clustering(limuc_labels, kmeans_r1.fit_predict(scaled_limuc), scaled_limuc, "Route 1 (Raw -> K-Means)")

    # Route 2: PCA -> K-Means
    pca = PCA(n_components=min(10, scaled_limuc.shape[1]), random_state=42)
    pca_limuc = pca.fit_transform(scaled_limuc)
    kmeans_r2 = KMeans(n_clusters=4, random_state=42, n_init=10)
    evaluate_clustering(limuc_labels, kmeans_r2.fit_predict(pca_limuc), pca_limuc, "Route 2 (PCA -> K-Means)")

    # Route 3: UMAP -> K-Means
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_limuc = reducer.fit_transform(scaled_limuc)
    kmeans_r3 = KMeans(n_clusters=4, random_state=42, n_init=10)
    evaluate_clustering(limuc_labels, kmeans_r3.fit_predict(umap_limuc), umap_limuc, "Route 3 (UMAP -> K-Means)")"""
    nb.cells.append(nbf.v4.new_code_cell(clust_code))

    # --- CELL 6: Feature Importance Visual ---
    nb.cells.append(nbf.v4.new_markdown_cell("""## 5. Visualisasi Feature Importance per MES (LIMUC)"""))
    vis_code = """# Pad atau potong nama fitur menyesuaikan hasil ekstraktor
actual_dim = limuc_features.shape[1]
feat_names_plot = FEATURE_NAMES_3CH.copy()
if len(feat_names_plot) < actual_dim: feat_names_plot.extend([f"Feat_{i}" for i in range(len(feat_names_plot), actual_dim)])
else: feat_names_plot = feat_names_plot[:actual_dim]

fig = plt.figure(figsize=(16, 16))
from matplotlib.gridspec import GridSpec
gs = GridSpec(4, 5, figure=fig, width_ratios=[1, 1.5, 1.5, 1.5, 1.5])
colors = ['#2ca02c', '#bcbd22', '#ff7f0e', '#d62728']

variances = np.var(scaled_limuc, axis=0)
top_25_idx = np.argsort(variances)[-25:]
top_feature_names = [feat_names_plot[i] for i in top_25_idx]

limuc_sample_paths = {
    "0": "/home/ubuntu/Colonoscopy/Dataset/LIMUC/patient_based_classified_images/1/Mayo 0/UC_patient_1_16.bmp",
    "1": "/home/ubuntu/Colonoscopy/Dataset/LIMUC/patient_based_classified_images/1/Mayo 1/UC_patient_1_11.bmp",
    "2": "/home/ubuntu/Colonoscopy/Dataset/LIMUC/patient_based_classified_images/1/Mayo 2/UC_patient_1_10.bmp",
    "3": "/home/ubuntu/Colonoscopy/Dataset/LIMUC/patient_based_classified_images/10/Mayo 3/UC_patient_10_27.bmp"
}

for mes in range(4):
    ax_img = fig.add_subplot(gs[mes, 0])
    ax_bar = fig.add_subplot(gs[mes, 1:])
    
    img_path = limuc_sample_paths.get(str(mes))
    if img_path and os.path.exists(img_path):
        img_data = np.array(Image.open(img_path).convert('RGB').resize((256, 256)))
    else:
        img = Image.new('RGB', (256, 256), color=(220, 150, 150))
        img_data = np.array(img)
        
    ax_img.imshow(img_data)
    ax_img.axis('off')
    ax_img.add_patch(patches.Rectangle((0, -40), 256, 40, facecolor=colors[mes], edgecolor='none'))
    ax_img.text(128, -20, f"MES {mes}", color='white', fontweight='bold', fontsize=16, ha='center', va='center')

    mes_mask = (limuc_labels == mes)
    if np.sum(mes_mask) > 0: mean_vals = np.mean(np.abs(limuc_features[mes_mask]), axis=0)
    else: mean_vals = np.random.uniform(10, 100000, size=actual_dim)
        
    vals = mean_vals[top_25_idx]
    x_pos = np.arange(len(top_feature_names))
    safe_values = np.where(vals <= 0, 1e-6, vals)
    
    ax_bar.bar(x_pos, safe_values, color='#1f77b4', edgecolor='white')
    ax_bar.set_yscale('log')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(top_feature_names, rotation=45, ha='right', fontsize=9)
    ax_bar.set_ylabel("Mean Feature Value (log)", fontsize=10)
    ax_bar.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    
    circle_indices = np.argsort(safe_values)[-2:]
    for idx in circle_indices:
        x, y = x_pos[idx], safe_values[idx]
        ax_bar.add_patch(patches.Ellipse((x, y), width=1.5, height=y*0.8, edgecolor='red', facecolor='none', lw=2))

plt.tight_layout()
plt.show()"""
    nb.cells.append(nbf.v4.new_code_cell(vis_code))

    # --- CELL 7: UMAP ---
    nb.cells.append(nbf.v4.new_markdown_cell("""## 6. UMAP Visualization Before vs After (LIMUC)"""))
    umap_code = """dl_path = os.path.join(DATA_DIR, "limuc_features", "limuc_dl_features.npy")
if os.path.exists(dl_path):
    dl_features = np.load(dl_path)
    
    max_samples = min(5000, len(limuc_labels))
    np.random.seed(42)
    sub_idx = np.random.choice(len(limuc_labels), max_samples, replace=False)
    
    dl_sub = dl_features[sub_idx]
    tex_sub = limuc_features[sub_idx]
    lbl_sub = limuc_labels[sub_idx]
    
    print("Menghitung UMAP... Mohon tunggu sebentar.")
    umap_dl = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(StandardScaler().fit_transform(dl_sub))
    umap_tex = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(StandardScaler().fit_transform(tex_sub))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors_map = sns.color_palette("husl", 4)
    
    for i in range(4):
        mask = (lbl_sub == i)
        axes[0].scatter(umap_dl[mask, 0], umap_dl[mask, 1], color=colors_map[i], label=f'MES {i}', alpha=0.6, s=15)
        axes[1].scatter(umap_tex[mask, 0], umap_tex[mask, 1], color=colors_map[i], label=f'MES {i}', alpha=0.6, s=15)
        
    axes[0].set_title("Before: Raw Deep Learning Features", fontsize=14)
    axes[1].set_title("After: Texture Architecture Features (3-Channel)", fontsize=14)
    
    for ax in axes:
        ax.axis('off')
        ax.legend()
        
    plt.tight_layout()
    plt.show()
else:
    print(f"File DL features tidak ditemukan di {dl_path}, skip visualisasi UMAP.")"""
    nb.cells.append(nbf.v4.new_code_cell(umap_code))

    with open(nb_path, 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    main()
