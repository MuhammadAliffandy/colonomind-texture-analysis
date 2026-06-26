import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
# Menggunakan backend non-interactive 'Agg' agar tidak error di server DGX (Headless)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import umap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont
import pywt
import scipy.stats
from pathlib import Path
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("radiomics").setLevel(logging.ERROR)

# Add root project dir to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import SimpleITK as sitk
    from radiomics import glcm
    RADIOMICS_AVAILABLE = True
except ImportError:
    sitk = None
    glcm = None
    RADIOMICS_AVAILABLE = False
    print("[WARNING] PyRadiomics tidak terinstal. Pastikan untuk menginstal pyradiomics jika Anda butuh fitur GLCM.")

# ==========================================
# KONFIGURASI DIREKTORI UNTUK SERVER DGX (D13K48009)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
FIG_DIR = REPORTS_DIR / "figures"

os.makedirs(FIG_DIR, exist_ok=True)
for d in ['limuc_features', 'tmc_features']:
    os.makedirs(DATA_DIR / d, exist_ok=True)

# Direktori Gambar Mentah (Raw Image Dataset) di Server DGX D13K48009
LIMUC_RAW_DIR = Path("/raid/D13K48009/texture/LIMUC")
TMC_RAW_DIR = Path("/raid/D13K48009/texture/TMC")

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

def _smart_preprocess(img):
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
        
    return np.array(all_features, dtype=np.float32)

def evaluate_clustering(labels_true, labels_pred, features, route_name):
    ari = adjusted_rand_score(labels_true, labels_pred)
    if len(features) > 15000:
        idx = np.random.choice(len(features), 15000, replace=False)
        sil = silhouette_score(features[idx], labels_pred[idx])
    else:
        sil = silhouette_score(features, labels_pred)
    print(f"{route_name:25} | ARI: {ari:+.4f} | Silhouette: {sil:+.4f}")

def generate_threshold_table(features, labels, target_feats, dataset_name):
    if len(features) == 0: return None
    
    indices = []
    headers = []
    for f in target_feats:
        if f in FEATURE_NAMES_3CH:
            indices.append(FEATURE_NAMES_3CH.index(f))
            headers.append(f"{f} Range (Avg)")
        else:
            indices.append(-1)
            headers.append(f)
            
    table_data = []
    for mes in range(4):
        mask = (labels == mes)
        if np.sum(mask) == 0: continue
            
        row = [f"MES {mes}"]
        for idx in indices:
            if idx != -1:
                vals = features[mask, idx]
                q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
                avg = np.mean(vals)
                row.append(f"{q1:.4f} - {q3:.4f} ({avg:.4f})")
            else:
                row.append("N/A")
        table_data.append(row)
        
    df = pd.DataFrame(table_data, columns=["MES Level"] + headers)
    # Save as CSV di DGX
    csv_path = os.path.join(REPORTS_DIR, f"{dataset_name.lower()}_thresholds.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Disimpan Threshold Table {dataset_name} ke: {csv_path}")
    return df

def main():
    print("✅ Modul dan Direktori siap.")
    
    # ---------------------------------------------------------
    # 2. Ekstraksi Dataset LIMUC (Patient Level)
    # ---------------------------------------------------------
    limuc_tex_path = DATA_DIR / "limuc_features" / "limuc_texture_features_3ch_v2.npy"
    limuc_lbl_path = DATA_DIR / "limuc_features" / "limuc_labels_3ch_v2.npy"

    limuc_features = np.array([])
    limuc_labels = np.array([])
    if limuc_tex_path.exists() and limuc_lbl_path.exists():
        print("✅ File ekstraksi LIMUC sudah ditemukan. Melewati proses ekstraksi...")
        limuc_features = np.load(limuc_tex_path)
        limuc_labels = np.load(limuc_lbl_path)
        print(f"Loaded LIMUC: {limuc_features.shape}")
    else:
        print("⏳ Menjalankan ekstraksi LIMUC secara Patient-Level...")
        base_target = LIMUC_RAW_DIR / "patient_based_classified_images"
        if not base_target.exists(): base_target = LIMUC_RAW_DIR # Fallback
        
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
        
        for idx in tqdm(range(len(limuc_img_paths)), desc="LIMUC Extraction"):
            img = cv2.imread(limuc_img_paths[idx])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_prep = _smart_preprocess(img)
                feats = extract_3ch_features(img_prep)
                limuc_features_list.append(feats)
                limuc_labels_list.append(limuc_img_labels[idx])
                
        if len(limuc_features_list) > 0:
            limuc_features = np.array(limuc_features_list)
            limuc_labels = np.array(limuc_labels_list)
            np.save(limuc_tex_path, limuc_features)
            np.save(limuc_lbl_path, limuc_labels)
            print(f"✅ Ekstraksi selesai! Shape: {limuc_features.shape}. Disimpan ke {limuc_tex_path}")

    # ---------------------------------------------------------
    # 3. Ekstraksi Dataset TMC-UCM
    # ---------------------------------------------------------
    tmc_tex_path = DATA_DIR / "tmc_features" / "tmc_texture_features_3ch_v3.npy"
    tmc_lbl_path = DATA_DIR / "tmc_features" / "tmc_labels_3ch_v3.npy"

    tmc_features = np.array([])
    tmc_labels = np.array([])
    if tmc_tex_path.exists() and tmc_lbl_path.exists():
        print("✅ File ekstraksi TMC sudah ditemukan. Melewati proses ekstraksi...")
        tmc_features = np.load(tmc_tex_path)
        tmc_labels = np.load(tmc_lbl_path)
        print(f"Loaded TMC: {tmc_features.shape}")
    else:
        print("⏳ Menjalankan ekstraksi TMC...")
        
        # DGX parser from train.txt and test.txt (sesuai konvensi dgx_extract_tmc.py)
        tmc_img_paths = []
        tmc_img_labels = []
        
        def parse_tmc_split(txt_file):
            paths, lbls = [], []
            if txt_file.exists():
                with open(txt_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        parts = line.split(",") if "," in line else line.split()
                        if len(parts) >= 2:
                            img_name_raw = parts[0].strip()
                            label = int(float(parts[1].strip()))
                            basename = os.path.basename(img_name_raw)
                            if "augment/" in img_name_raw:
                                abs_path = TMC_RAW_DIR / "augment" / basename
                            else:
                                abs_path = TMC_RAW_DIR / "images" / basename
                            if abs_path.exists():
                                paths.append(str(abs_path))
                                lbls.append(label)
            return paths, lbls
        
        train_imgs, train_lbls = parse_tmc_split(TMC_RAW_DIR / "train.txt")
        test_imgs, test_lbls = parse_tmc_split(TMC_RAW_DIR / "test.txt")
        tmc_img_paths = train_imgs + test_imgs
        tmc_img_labels = train_lbls + test_lbls
        
        # Fallback if text files don't exist
        if len(tmc_img_paths) == 0:
            print("[INFO] Fallback TMC parsing from directory rglob...")
            for img_path in TMC_RAW_DIR.rglob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.bmp']:
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
                
        if len(tmc_features_list) > 0:
            tmc_features = np.array(tmc_features_list)
            tmc_labels = np.array(tmc_labels_list)
            np.save(tmc_tex_path, tmc_features)
            np.save(tmc_lbl_path, tmc_labels)
            print(f"✅ Ekstraksi selesai! Shape: {tmc_features.shape}")

    # ---------------------------------------------------------
    # 4. Evaluasi Clustering (LIMUC)
    # ---------------------------------------------------------
    from matplotlib.gridspec import GridSpec
    if len(limuc_features) > 0:
        print("\n[LIMUC] Z-score standardization...")
        scaler = StandardScaler()
        scaled_limuc = scaler.fit_transform(limuc_features)

        print("Executing 3 Clustering Routes for LIMUC...")
        kmeans_r1 = KMeans(n_clusters=4, random_state=42, n_init=10)
        evaluate_clustering(limuc_labels, kmeans_r1.fit_predict(scaled_limuc), scaled_limuc, "Route 1 (Raw -> K-Means)")

        pca = PCA(n_components=min(10, scaled_limuc.shape[1]), random_state=42)
        pca_limuc = pca.fit_transform(scaled_limuc)
        kmeans_r2 = KMeans(n_clusters=4, random_state=42, n_init=10)
        evaluate_clustering(limuc_labels, kmeans_r2.fit_predict(pca_limuc), pca_limuc, "Route 2 (PCA -> K-Means)")

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        umap_limuc = reducer.fit_transform(scaled_limuc)
        kmeans_r3 = KMeans(n_clusters=4, random_state=42, n_init=10)
        evaluate_clustering(limuc_labels, kmeans_r3.fit_predict(umap_limuc), umap_limuc, "Route 3 (UMAP -> K-Means)")

        # 5. Visualisasi Feature Importance per MES (LIMUC)
        actual_dim = limuc_features.shape[1]
        feat_names_plot = FEATURE_NAMES_3CH.copy()
        if len(feat_names_plot) < actual_dim: feat_names_plot.extend([f"Feat_{i}" for i in range(len(feat_names_plot), actual_dim)])
        else: feat_names_plot = feat_names_plot[:actual_dim]

        fig = plt.figure(figsize=(16, 16))
        gs = GridSpec(4, 5, figure=fig, width_ratios=[1, 1.5, 1.5, 1.5, 1.5])
        colors = ['#2ca02c', '#bcbd22', '#ff7f0e', '#d62728']

        variances = np.var(scaled_limuc, axis=0)
        top_25_idx = np.argsort(variances)[-25:]
        top_feature_names = [feat_names_plot[i] for i in top_25_idx]

        limuc_sample_paths = {
            "0": f"{LIMUC_RAW_DIR}/patient_based_classified_images/1/Mayo 0/UC_patient_1_16.bmp",
            "1": f"{LIMUC_RAW_DIR}/patient_based_classified_images/1/Mayo 1/UC_patient_1_11.bmp",
            "2": f"{LIMUC_RAW_DIR}/patient_based_classified_images/1/Mayo 2/UC_patient_1_10.bmp",
            "3": f"{LIMUC_RAW_DIR}/patient_based_classified_images/10/Mayo 3/UC_patient_10_27.bmp"
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
            if np.sum(mes_mask) > 0: mean_vals = np.mean(scaled_limuc[mes_mask], axis=0)
            else: mean_vals = np.zeros(actual_dim)
                
            vals = mean_vals[top_25_idx]
            x_pos = np.arange(len(top_feature_names))
            
            colors_bar = ['#d62728' if v > 0 else '#1f77b4' for v in vals]
            ax_bar.bar(x_pos, vals, color=colors_bar, edgecolor='white')
            ax_bar.axhline(0, color='black', linewidth=1)
            ax_bar.set_xticks(x_pos)
            ax_bar.set_xticklabels(top_feature_names, rotation=45, ha='right', fontsize=9)
            ax_bar.set_ylabel("Standardized Value (Z-Score)", fontsize=10)
            ax_bar.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            
            circle_indices = np.argsort(np.abs(vals))[-2:]
            for idx in circle_indices:
                x, y = x_pos[idx], vals[idx]
                offset = 0.1 if y > 0 else -0.1
                ax_bar.add_patch(patches.Ellipse((x, y + offset), width=1.5, height=np.abs(y)*0.4 + 0.1, edgecolor='red', facecolor='none', lw=2))

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "limuc_feature_importance.png"))
        print(f"✅ Disimpan: {os.path.join(FIG_DIR, 'limuc_feature_importance.png')}")
        plt.close()

        # 6. UMAP Visualization Before vs After (LIMUC)
        dl_path = DATA_DIR / "limuc_features" / "limuc_dl_features.npy"
        if dl_path.exists():
            dl_features = np.load(dl_path)
            max_samples = min(5000, len(limuc_labels))
            np.random.seed(42)
            sub_idx = np.random.choice(len(limuc_labels), max_samples, replace=False)
            
            dl_sub = dl_features[sub_idx]
            tex_sub = limuc_features[sub_idx]
            lbl_sub = limuc_labels[sub_idx]
            
            print("[LIMUC] Menghitung UMAP... Mohon tunggu sebentar.")
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
            plt.savefig(os.path.join(FIG_DIR, "limuc_umap_comparison.png"))
            print(f"✅ Disimpan: {os.path.join(FIG_DIR, 'limuc_umap_comparison.png')}")
            plt.close()

    # ---------------------------------------------------------
    # 7. Evaluasi Clustering (TMC)
    # ---------------------------------------------------------
    if len(tmc_features) > 0:
        print("\n[TMC] Z-score standardization untuk TMC...")
        scaler_tmc = StandardScaler()
        scaled_tmc = scaler_tmc.fit_transform(tmc_features)

        print("Executing 3 Clustering Routes for TMC...")
        kmeans_t1 = KMeans(n_clusters=4, random_state=42, n_init=10)
        evaluate_clustering(tmc_labels, kmeans_t1.fit_predict(scaled_tmc), scaled_tmc, "Route 1 (Raw -> K-Means)")

        pca_t = PCA(n_components=min(10, scaled_tmc.shape[1]), random_state=42)
        pca_tmc = pca_t.fit_transform(scaled_tmc)
        kmeans_t2 = KMeans(n_clusters=4, random_state=42, n_init=10)
        evaluate_clustering(tmc_labels, kmeans_t2.fit_predict(pca_tmc), pca_tmc, "Route 2 (PCA -> K-Means)")

        reducer_t = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        umap_tmc = reducer_t.fit_transform(scaled_tmc)
        kmeans_t3 = KMeans(n_clusters=4, random_state=42, n_init=10)
        evaluate_clustering(tmc_labels, kmeans_t3.fit_predict(umap_tmc), umap_tmc, "Route 3 (UMAP -> K-Means)")

        # 8. Visualisasi Feature Importance per MES (TMC)
        actual_dim_tmc = tmc_features.shape[1]
        fig = plt.figure(figsize=(16, 16))
        gs = GridSpec(4, 5, figure=fig, width_ratios=[1, 1.5, 1.5, 1.5, 1.5])
        
        plot_scaler_tmc = MinMaxScaler(feature_range=(1, 100))
        plot_tmc = plot_scaler_tmc.fit_transform(tmc_features)
        variances_tmc = np.var(plot_tmc, axis=0)
        top_25_idx_t = np.argsort(variances_tmc)[-25:]
        top_feature_names_t = [feat_names_plot[i] for i in top_25_idx_t]

        tmc_sample_paths = {
            "0": f"{TMC_RAW_DIR}/images/P02264.JPG",
            "1": f"{TMC_RAW_DIR}/images/P04880.JPG",
            "2": f"{TMC_RAW_DIR}/images/P07855.JPG",
            "3": f"{TMC_RAW_DIR}/images/P10984.JPG"
        }

        colors = ['#2ca02c', '#bcbd22', '#ff7f0e', '#d62728']

        for mes in range(4):
            ax_img = fig.add_subplot(gs[mes, 0])
            ax_bar = fig.add_subplot(gs[mes, 1:])
            
            img_path = tmc_sample_paths.get(str(mes))
            if img_path and os.path.exists(img_path):
                img_data = np.array(Image.open(img_path).convert('RGB').resize((256, 256)))
            else:
                img = Image.new('RGB', (256, 256), color=(150, 150, 220))
                img_data = np.array(img)
                
            ax_img.imshow(img_data)
            ax_img.axis('off')
            ax_img.add_patch(patches.Rectangle((0, -40), 256, 40, facecolor=colors[mes], edgecolor='none'))
            ax_img.text(128, -20, f"MES {mes}", color='white', fontweight='bold', fontsize=16, ha='center', va='center')

            mes_mask_t = (tmc_labels == mes)
            if np.sum(mes_mask_t) > 0: mean_vals_t = np.mean(plot_tmc[mes_mask_t], axis=0)
            else: mean_vals_t = np.zeros(actual_dim_tmc)
                
            vals_t = mean_vals_t[top_25_idx_t]
            x_pos_t = np.arange(len(top_feature_names_t))
            
            ax_bar.bar(x_pos_t, vals_t, color='#1f77b4', edgecolor='white')
            
            ax_bar.set_xticks(x_pos_t)
            ax_bar.set_xticklabels(top_feature_names_t, rotation=45, ha='right', fontsize=9)
            ax_bar.set_ylabel("Relative Feature Strength (0-100)", fontsize=10)
            ax_bar.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            
            circle_indices_t = np.argsort(np.abs(vals_t))[-2:]
            for idx in circle_indices_t:
                x, y = x_pos_t[idx], vals_t[idx]
                ax_bar.add_patch(patches.Ellipse((x, y + 2), width=1.5, height=y*0.15 + 2, edgecolor='red', facecolor='none', lw=2))

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "tmc_feature_importance.png"))
        print(f"✅ Disimpan: {os.path.join(FIG_DIR, 'tmc_feature_importance.png')}")
        plt.close()

        # 9. UMAP Visualization Before vs After (TMC)
        dl_path_tmc = DATA_DIR / "tmc_features" / "tmc_dl_features.npy"
        if dl_path_tmc.exists():
            dl_features_tmc = np.load(dl_path_tmc)
            max_samples = min(5000, len(tmc_labels))
            np.random.seed(42)
            sub_idx = np.random.choice(len(tmc_labels), max_samples, replace=False)
            
            dl_sub_t = dl_features_tmc[sub_idx]
            tex_sub_t = tmc_features[sub_idx]
            lbl_sub_t = tmc_labels[sub_idx]
            
            print("[TMC] Menghitung UMAP TMC... Mohon tunggu sebentar.")
            umap_dl_t = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(StandardScaler().fit_transform(dl_sub_t))
            umap_tex_t = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(StandardScaler().fit_transform(tex_sub_t))
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            colors_map = sns.color_palette("husl", 4)
            for i in range(4):
                mask = (lbl_sub_t == i)
                axes[0].scatter(umap_dl_t[mask, 0], umap_dl_t[mask, 1], color=colors_map[i], label=f'MES {i}', alpha=0.6, s=15)
                axes[1].scatter(umap_tex_t[mask, 0], umap_tex_t[mask, 1], color=colors_map[i], label=f'MES {i}', alpha=0.6, s=15)
                
            axes[0].set_title("TMC Before: Raw Deep Learning Features", fontsize=14)
            axes[1].set_title("TMC After: Texture Architecture Features (3-Channel)", fontsize=14)
            
            for ax in axes:
                ax.axis('off')
                ax.legend()
                
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, "tmc_umap_comparison.png"))
            print(f"✅ Disimpan: {os.path.join(FIG_DIR, 'tmc_umap_comparison.png')}")
            plt.close()

    # ---------------------------------------------------------
    # 10. Advanced Visualizations (Scatter & Box Plots)
    # ---------------------------------------------------------
    colors_map = sns.color_palette("husl", 4)
    label_map = {0: "Mayo 0 (Healthy)", 1: "Mayo 1 (Mild)", 2: "Mayo 2 (Moderate)", 3: "Mayo 3 (Severe)"}
    
    if 'Green_GLCM_Contrast' in FEATURE_NAMES_3CH:
        idx_contrast = FEATURE_NAMES_3CH.index('Green_GLCM_Contrast')
        idx_ll_mean = FEATURE_NAMES_3CH.index('Green_LL_Mean')
        idx_hh_var = FEATURE_NAMES_3CH.index('Green_HH_Var')
        
        if len(limuc_features) > 0:
            # SCATTER PLOT
            plt.figure(figsize=(8, 6))
            for mes in range(4):
                mask = (limuc_labels == mes)
                x_vals = limuc_features[mask, idx_contrast]
                y_vals = limuc_features[mask, idx_ll_mean]
                lbl = label_map[mes]
                plt.scatter(x_vals, y_vals, c=[colors_map[mes]], label=lbl, alpha=0.7, s=20)
                
            plt.title("LIMUC: GLCM Contrast vs DWT LL Mean", fontsize=14, fontweight='bold')
            plt.xlabel("Texture Roughness (GLCM Contrast)", fontsize=12)
            plt.ylabel("Image Brightness (DWT LL Mean)", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, "limuc_scatter_contrast_vs_mean.png"))
            plt.close()

            # BOX PLOT
            plt.figure(figsize=(8, 6))
            df_box = pd.DataFrame({'DWT HH Variance': limuc_features[:, idx_hh_var], 'Class': limuc_labels})
            df_box['Class'] = df_box['Class'].map(label_map)
            sns.boxplot(x='Class', y='DWT HH Variance', data=df_box, palette="husl", showfliers=False)
            plt.title("LIMUC: Edge Sharpness (DWT HH Variance)", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, "limuc_boxplot_hh_variance.png"))
            plt.close()

        if len(tmc_features) > 0:
            # SCATTER PLOT TMC
            plt.figure(figsize=(8, 6))
            for mes in range(4):
                mask_t = (tmc_labels == mes)
                x_vals_t = tmc_features[mask_t, idx_contrast]
                y_vals_t = tmc_features[mask_t, idx_ll_mean]
                lbl = label_map[mes]
                plt.scatter(x_vals_t, y_vals_t, c=[colors_map[mes]], label=lbl, alpha=0.7, s=20)
                
            plt.title("TMC: GLCM Contrast vs DWT LL Mean", fontsize=14, fontweight='bold')
            plt.xlabel("Texture Roughness (GLCM Contrast)", fontsize=12)
            plt.ylabel("Image Brightness (DWT LL Mean)", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, "tmc_scatter_contrast_vs_mean.png"))
            plt.close()

            # BOX PLOT TMC
            plt.figure(figsize=(8, 6))
            df_box_t = pd.DataFrame({'DWT HH Variance': tmc_features[:, idx_hh_var], 'Class': tmc_labels})
            df_box_t['Class'] = df_box_t['Class'].map(label_map)
            sns.boxplot(x='Class', y='DWT HH Variance', data=df_box_t, palette="husl", showfliers=False)
            plt.title("TMC: Edge Sharpness (DWT HH Variance)", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, "tmc_boxplot_hh_variance.png"))
            plt.close()

    # ---------------------------------------------------------
    # 11. Rule-Based Thresholds
    # ---------------------------------------------------------
    generate_threshold_table(limuc_features, limuc_labels, ['Green_HH_Ent', 'Green_HL_Mean', 'Green_LH_Mean'], "LIMUC")
    generate_threshold_table(tmc_features, tmc_labels, ['Green_HL_Ent', 'Green_LH_Ent', 'Green_GLCM_Dissimilarity'], "TMC")
    
    print("\n🎉 Eksekusi Selesai! Semua grafik dan tabel thresholds telah disimpan di folder:", REPORTS_DIR)

if __name__ == "__main__":
    main()
