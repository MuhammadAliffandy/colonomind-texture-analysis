import os
# MUST be set before importing tensorflow to use Keras 2 compatible mode
# This fixes the 'SlicingOpLambda' error when loading old .h5 models
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path

# Add root project dir to python path so we can import from app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm
from PIL import Image
from app.texture_extractor import extract_glcm_dwt, _smart_preprocess

# ---------------------------------------------------------------------------
# Setup & Config
# ---------------------------------------------------------------------------
# Set path dataset TMC
TMC_ROOT = Path("/raid/D13K48009/texture/TMC")

if not TMC_ROOT.exists():
    print(f"[ERROR] Direktori TMC tidak ditemukan di {TMC_ROOT}")
    print("Pastikan script ini dijalankan di server DGX dengan path yang sesuai!")
    sys.exit(1)

# Default Model Paths
MODEL_DIR = Path(__file__).resolve().parent.parent / "Model-colono"
MODEL_PATH = MODEL_DIR / "models-TryFindingBestModel.h5"
SCALER_PATH = MODEL_DIR / "models-scaler_handcrafted_20.pkl"
UMAP_PATH = MODEL_DIR / "models-umap_model_mixed.pkl"

if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not UMAP_PATH.exists():
    print(f"[ERROR] Keras model, Scaler, atau UMAP tidak ditemukan di {MODEL_DIR}")
    sys.exit(1)

print("[INFO] Loading Models...")

# 1. Models
full_model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
feature_layer = full_model.get_layer("dense_5")
feature_extractor = tf.keras.Model(
    inputs=full_model.input,
    outputs=feature_layer.output,
    name="feature_extractor"
)

scaler = joblib.load(str(SCALER_PATH))
umap_model = joblib.load(str(UMAP_PATH))

# ---------------------------------------------------------------------------
# Parser Text File
# ---------------------------------------------------------------------------
def parse_tmc_split(txt_file):
    """
    Parse test.txt atau train.txt
    Bisa handle format: 'filename.jpg 0' atau 'images/filename.jpg, 1'
    """
    image_paths = []
    labels = []
    
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Coba pisahkan dengan koma atau spasi
            parts = line.split(",") if "," in line else line.split()
            
            if len(parts) >= 2:
                img_name_raw = parts[0].strip()
                label = int(float(parts[1].strip()))
                
                # Ekstrak nama file saja karena txt aslinya menggunakan absolute path dari mesin lain
                # misal: /home/zsj/dataset/TMC-UCM/images/P09565.JPG -> P09565.JPG
                import os
                basename = os.path.basename(img_name_raw)
                
                # Resolusi absolute path ke folder images di TMC_ROOT
                # Jika aslinya dari folder augment, kita akan cari di augment/ juga
                if "augment/" in img_name_raw:
                    abs_path = TMC_ROOT / "augment" / basename
                else:
                    abs_path = TMC_ROOT / "images" / basename
                
                if abs_path.exists():
                    image_paths.append(abs_path)
                    labels.append(label)
                else:
                    print(f"[WARNING] Image missing: {abs_path}")
            
    return image_paths, labels

print("[INFO] Parsing Dataset TMC...")
train_imgs, train_lbls = parse_tmc_split(TMC_ROOT / "train.txt")
test_imgs, test_lbls = parse_tmc_split(TMC_ROOT / "test.txt")

all_img_paths = train_imgs + test_imgs
all_labels = train_lbls + test_lbls

total_images = len(all_img_paths)
if total_images == 0:
    print("[ERROR] Tidak ada gambar yang berhasil di-parse dari file txt!")
    sys.exit(1)
    
print(f"[INFO] Ditemukan {len(train_imgs)} Train dan {len(test_imgs)} Test. Total = {total_images} gambar.")

# ---------------------------------------------------------------------------
# Extraction Loop
# ---------------------------------------------------------------------------
all_texture_feats = []
all_dl_feats = []
final_labels = []

print(f"[INFO] Memulai Ekstraksi (Deep Features & 20-Handcrafted) via GPU...")

for i, img_path in enumerate(tqdm(all_img_paths)):
    try:
        # Load Image
        pil_img = Image.open(str(img_path)).convert("RGB")
        rgb_image = np.array(pil_img)
        
        # Preprocess
        processed_img = _smart_preprocess(rgb_image)
        input_tensor = np.expand_dims(processed_img.astype(np.float32) / 255.0, axis=0)
        
        # 1. Ekstraksi Handcrafted (20 Fitur)
        texture_feat = extract_glcm_dwt(processed_img)
        
        # 2. Skala dan UMAP
        scaled_feat = scaler.transform(np.array(texture_feat).reshape(1, -1))
        umap_feat = umap_model.transform(scaled_feat)
        
        # 3. Ekstraksi Deep Features (Multimodal 3-input)
        keras_inputs = [input_tensor, scaled_feat, umap_feat]
        dl_feat = feature_extractor.predict(keras_inputs, verbose=0)[0]
        
        all_texture_feats.append(texture_feat)
        all_dl_feats.append(dl_feat)
        final_labels.append(all_labels[i])

    except Exception as e:
        print(f"[ERROR] Failed extracting {img_path}: {e}")
        continue

print(f"[INFO] Saving results...")
np.save("tmc_dl_features.npy", np.array(all_dl_feats))
np.save("tmc_texture_features.npy", np.array(all_texture_feats))
np.save("tmc_labels.npy", np.array(final_labels))

print("[INFO] Features saved for analysis!")
